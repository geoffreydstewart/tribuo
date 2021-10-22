/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tribuo.clustering.hdbscan;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.*;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ImmutableClusteringInfo;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.*;
import java.util.Map.Entry;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An HDBSCAN* trainer. Update all the rest of this!
 * <p>
 * It's slightly contorted to fit the Tribuo Trainer and Model API, as the cluster assignments
 * can only be retrieved from the model after training, and require re-evaluating each example.
 * <p>
 * The Trainer has a parameterised distance function, and a selectable number
 * of threads used in the training step. The thread pool is local to an invocation of train,
 * so there can be multiple concurrent trainings.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 * <p>
 * For more on optional kmeans++ initialisation, see:
 * <pre>
 * D. Arthur, S. Vassilvitskii.
 * "K-Means++: The Advantages of Careful Seeding"
 * <a href="https://theory.stanford.edu/~sergei/papers/kMeansPP-soda">PDF</a>
 * </pre>
 */
public class HdbscanTrainer implements Trainer<ClusterID> {
    private static final Logger logger = Logger.getLogger(HdbscanTrainer.class.getName());

    static final int OUTLIER_NOISE_CLUSTER_LABEL = 0;

    // The fractional value to apply to the size of the dataset to determine the number of exemplars.
    private static final double NUM_EXEMPLARS_FRACTIONAL_VALUE = 0.01;

    // The maximum number of exemplars to use for prediction.
    private static final int MAX_NUM_EXEMPLARS = 100;

    /**
     * Available distance functions.
     */
    public enum Distance {
        /**
         * Euclidean (or l2) distance.
         */
        EUCLIDEAN,
        /**
         * Cosine similarity as a distance measure.
         */
        COSINE,
        /**
         * L1 (or Manhattan) distance.
         */
        L1
    }

    @Config(mandatory = true, description = "The minimum number of points required to form a cluster.")
    private int minClusterSize;

    @Config(description = "The distance function to use.")
    private Distance distanceType;

    @Config(description = "The number of nearest-neighbors to use in the initial density approximation.")
    private int k;

    @Config(description = "The number of threads to use for training.")
    private int numThreads = 1;

    private int trainInvocationCounter;

    /**
     * for olcut.
     */
    private HdbscanTrainer() {
    }

    /**
     * Constructs an HDBSCAN* trainer with only the minClusterSize parameter.
     *
     * @param minClusterSize The minimum number of points required to form a cluster.
     */
    public HdbscanTrainer(int minClusterSize) {
        this(minClusterSize, Distance.EUCLIDEAN, minClusterSize, 1);
    }

    /**
     * Constructs an HDBSCAN* trainer using the supplied parameters.
     *
     * @param minClusterSize The minimum number of points required to form a cluster.
     * @param distanceType The distance function.
     * @param k The number of nearest-neighbors to use in the initial density approximation.
     * @param numThreads The number of threads.
     */
    public HdbscanTrainer(int minClusterSize, Distance distanceType, int k, int numThreads) {
        this.minClusterSize = minClusterSize;
        this.distanceType = distanceType;
        this.k = k;
        this.numThreads = numThreads;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        // no post config steps required at this time
    }

    @Override
    public HdbscanModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance) {
        // increment the invocation count.
        TrainerProvenance trainerProvenance;
        synchronized (this) {
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();

        //ForkJoinPool fjp = new ForkJoinPool(numThreads);

        SGDVector[] data = new DenseVector[examples.size()];
        int n = 0;
        for (Example<ClusterID> example : examples) {
            data[n] = DenseVector.createDenseVector(example, featureMap, false);
            n++;
        }

        DenseVector coreDistances = calculateCoreDistances(data, k, distanceType);
        ExtendedMinimumSpanningTree emst = constructMST(data, coreDistances, distanceType);

        double[] pointNoiseLevels = new double[data.length];    // The levels at which each point becomes noise
        int[] pointLastClusters = new int[data.length];         // The last label of each point before becoming noise
        Map<Integer, int[]> hierarchy = new HashMap<>();        // The HDBSCAN* hierarchy
        List<HdbscanCluster> clusters = computeHierarchyAndClusterTree(emst, minClusterSize, pointNoiseLevels,
                                                                       pointLastClusters, hierarchy);
        propagateTree(clusters);
        List<Integer> clusterLabels = findProminentClusters(hierarchy, clusters, data.length);
        DenseVector outlierScoresVector = calculateOutlierScores(pointNoiseLevels, pointLastClusters, clusters);
        Map<Integer, TreeMap<Double, Integer>> clusterAssignments =  generateClusterAssignments(clusterLabels, outlierScoresVector);

        // Use the cluster assignments, while they're available, to establish the clustering info
        Map<Integer, MutableLong> counts = new HashMap<>();
        for (Entry<Integer, TreeMap<Double, Integer>> e : clusterAssignments.entrySet()) {
            counts.put(e.getKey(), new MutableLong(e.getValue().size()));
        }
        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        // Compute the cluster exemplars. The TreeMaps in clusterAssignments are modified here, having nodes removed.
        List<ClusterExemplar> clusterExemplars = computeExemplars(data, clusterAssignments);

        logger.log(Level.INFO, "Hdbscan is done");

        ModelProvenance provenance = new ModelProvenance(HdbscanModel.class.getName(), OffsetDateTime.now(),
                examples.getProvenance(), trainerProvenance, runProvenance);

        return new HdbscanModel("", provenance, featureMap, outputMap, clusterLabels, outlierScoresVector,
                                clusterExemplars, distanceType);
    }

    @Override
    public HdbscanModel train(Dataset<ClusterID> dataset) {
        return train(dataset, Collections.emptyMap());
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    /**
     * Calculates the core distance for every point in the data set.
     *
     * @param data An array of {@link DenseVector} containing the data.
     * @param k The number of nearest-neighbors to use in these calculations.
     * @param distanceType The distance metric to employ.
     * @return A {@link DenseVector} containing the core distances for every point.
     */
    private static DenseVector calculateCoreDistances(SGDVector[] data, int k, Distance distanceType) {
        int numNeighbors = k - 1;
        DenseVector coreDistances = new DenseVector(data.length);

        if (numNeighbors == 0) {
            return coreDistances;
        }

        // TODO: this needs to be optimized through parallelization
        for (int point = 0; point < data.length; point++) {
            double[] kNNDistances = new double[numNeighbors];	//Sorted nearest distances found so far
            for (int i = 0; i < numNeighbors; i++) {
                kNNDistances[i] = Double.MAX_VALUE;
            }

            for (int neighbor = 0; neighbor < data.length; neighbor++) {
                if (point == neighbor)
                    continue;
                double distance = getDistance(data[point], data[neighbor], distanceType);

                //Check at which position in the nearest distances the current distance would fit:
                int neighborIndex = numNeighbors;
                while (neighborIndex >= 1 && distance < kNNDistances[neighborIndex-1]) {
                    neighborIndex--;
                }

                //Shift elements in the array to make room for the current distance:
                if (neighborIndex < numNeighbors) {
                    for (int shiftIndex = numNeighbors-1; shiftIndex > neighborIndex; shiftIndex--) {
                        kNNDistances[shiftIndex] = kNNDistances[shiftIndex-1];
                    }
                    kNNDistances[neighborIndex] = distance;
                }
            }
            // The core distance for the point is the distance to the furthest away neighbor
            coreDistances.set(point, kNNDistances[numNeighbors-1]);
        }
        return coreDistances;
    }

    /**
     * Constructs an extended minimum spanning tree of mutual reachability distances from the data, given the
     * core distances for each point.
     * @param data An array of {@link DenseVector} containing the data.
     * @param coreDistances A {@link DenseVector} containing the core distances for every point.
     * @param distanceType The distance metric to employ.
     * @return An {@link ExtendedMinimumSpanningTree} representation of the data using the mutual reachability distances,
     * and the graph is sorted by edge weight in ascending order.
     */
    private static ExtendedMinimumSpanningTree constructMST(SGDVector[] data, DenseVector coreDistances,
                                                            Distance distanceType) {
        // One bit is set (true) for each attached point, and unset (false) for unattached points:
        BitSet attachedPoints = new BitSet(data.length);

        // Each point has a current neighbor point in the tree, and a corresponding current neighbor's distance
        int[] nearestMRDNeighbors = new int[2*data.length - 1];
        double[] nearestMRDDistances = new double[2*data.length - 1];

        for (int i = 0; i < data.length-1; i++) {
            nearestMRDDistances[i] = Double.MAX_VALUE;
        }

        // The MST is expanded starting with the last point in the data set:
        int currentPoint = data.length-1;
        int numAttachedPoints = 1;
        attachedPoints.set(data.length-1);

        // Continue attaching points to the MST until all points are attached:
        while (numAttachedPoints < data.length) {
            int nearestMRDPoint = -1;
            double nearestMRDDistance = Double.MAX_VALUE;

            // Iterate through all unattached points, updating distances using the current point:
            for (int neighbor = 0; neighbor < data.length; neighbor++) {
                if (currentPoint == neighbor)
                    continue;
                if (attachedPoints.get(neighbor))
                    continue;

                double mutualReachabilityDistance = getDistance(data[currentPoint], data[neighbor], distanceType);
                if (coreDistances.get(currentPoint) > mutualReachabilityDistance)
                    mutualReachabilityDistance = coreDistances.get(currentPoint);
                if (coreDistances.get(neighbor) > mutualReachabilityDistance)
                    mutualReachabilityDistance = coreDistances.get(neighbor);

                if (mutualReachabilityDistance < nearestMRDDistances[neighbor]) {
                    nearestMRDDistances[neighbor] = mutualReachabilityDistance;
                    nearestMRDNeighbors[neighbor] = currentPoint;
                }

                // Check if the unattached point being updated is the closest to the tree:
                if (nearestMRDDistances[neighbor] <= nearestMRDDistance) {
                    nearestMRDDistance = nearestMRDDistances[neighbor];
                    nearestMRDPoint = neighbor;
                }
            }

            //Attach the closest point found in this iteration to the tree:
            attachedPoints.set(nearestMRDPoint);
            numAttachedPoints++;
            currentPoint = nearestMRDPoint;
        }

        // Create an array for vertices in the tree that each point attached to:
        int[] otherVertexIndices = new int[2*data.length - 1];
        for (int i = 0; i < data.length-1; i++) {
            otherVertexIndices[i] = i;
        }

        // Attach self edges:
        for (int i = data.length-1; i < data.length*2-1; i++) {
            int vertex = i - (data.length-1);
            nearestMRDNeighbors[i] = vertex;
            otherVertexIndices[i] = vertex;
            nearestMRDDistances[i] = coreDistances.get(vertex);
        }
        ExtendedMinimumSpanningTree emst = new ExtendedMinimumSpanningTree(data.length, nearestMRDNeighbors,
                                                                           otherVertexIndices, nearestMRDDistances);
        emst.quicksortByEdgeWeight();
        return emst;
    }

    /**
     * Computes the hierarchy and cluster tree from the extended minimum spanning tree. Additionally, the level
     * at which each point becomes noise is recorded.
     * @param emst The extended minimum spanning tree which has been sorted by edge weight in ascending order.
     * @param minClusterSize The minimum number of points which a cluster needs to be a valid cluster
     * @param pointNoiseLevels A double[] to be populated with the edge weight levels at which each point becomes noise
     * @param pointLastClusters An int[] to be populated with the last label of each point had before becoming noise
     * @param hierarchy The HDBSCAN* hierarchy.
     * @return The cluster tree.
     */
    private static List<HdbscanCluster> computeHierarchyAndClusterTree(ExtendedMinimumSpanningTree emst, int minClusterSize,
                                                                    double[] pointNoiseLevels, int[] pointLastClusters,
                                                                    Map<Integer, int[]> hierarchy) {

        int lineCount = 0; //Indicates the number of lines written into hierarchyFile.

        // The current edge being removed from the MST:
        int currentEdgeIndex = emst.getNumEdges()-1;

        int nextClusterLabel = 2;    // all points are currently assigned to cluster 1
        boolean nextLevelSignificant = true;

        // The current cluster numbers of each point in the data set:
        int[] currentClusterLabels = new int[emst.getNumVertices()];
        Arrays.fill(currentClusterLabels, 1);

        // A list of clusters in the cluster tree, with the 0th cluster (noise) null:
        ArrayList<HdbscanCluster> clusters = new ArrayList<>();
        clusters.add(null);
        clusters.add(new HdbscanCluster(1, null, Double.NaN, emst.getNumVertices()));

        // Sets for the clusters and vertices that are affected by the edge(s) being removed. The labels depend
        // on being ordered.
        TreeSet<Integer> affectedClusterLabels = new TreeSet<>();
        // there is no ordering requirement on the vertices
        HashSet<Integer> affectedVertices = new HashSet<>();

        while(currentEdgeIndex >= 0) {
            double currentEdgeWeight = emst.getEdgeWeightAtIndex(currentEdgeIndex);
            ArrayList<HdbscanCluster> newClusters = new ArrayList<>();

            // Remove all edges tied with the current edge weight, and store relevant clusters and vertices:
            while (currentEdgeIndex >= 0 && emst.getEdgeWeightAtIndex(currentEdgeIndex) == currentEdgeWeight){
                int firstVertex = emst.getFirstVertexAtIndex(currentEdgeIndex);
                int secondVertex = emst.getSecondVertexAtIndex(currentEdgeIndex);
                emst.getEdgeListForVertex(firstVertex).remove((Integer)secondVertex);
                emst.getEdgeListForVertex(secondVertex).remove((Integer)firstVertex);

                if (currentClusterLabels[firstVertex] == 0) {
                    currentEdgeIndex--;
                    continue;
                }

                affectedVertices.add(firstVertex);
                affectedVertices.add(secondVertex);
                affectedClusterLabels.add(currentClusterLabels[firstVertex]);
                currentEdgeIndex--;
            }

            if (affectedClusterLabels.isEmpty())
                continue;

            // Check each cluster affected for a possible split:
            while (!affectedClusterLabels.isEmpty()) {
                int examinedClusterLabel = affectedClusterLabels.last();
                affectedClusterLabels.remove(examinedClusterLabel);
                TreeSet<Integer> examinedVertices = new TreeSet<>();

                //Get all affected vertices that are members of the cluster currently being examined:
                Iterator<Integer> vertexIterator = affectedVertices.iterator();
                while (vertexIterator.hasNext()) {
                    int vertex = vertexIterator.next();

                    if (currentClusterLabels[vertex] == examinedClusterLabel) {
                        examinedVertices.add(vertex);
                        vertexIterator.remove();
                    }
                }

                TreeSet<Integer> firstChildCluster = null;
                LinkedList<Integer> unexploredFirstChildClusterPoints = null;
                int numChildClusters = 0;

                /*
                 * Check if the cluster has split or shrunk by exploring the graph from each affected
                 * vertex.  If there are two or more valid child clusters (each has >= minClusterSize
                 * points), the cluster has split.
                 * Note that firstChildCluster will only be fully explored if there is a cluster
                 * split, otherwise, only spurious components are fully explored, in order to label
                 * them noise.
                 */
                while (!examinedVertices.isEmpty()) {
                    TreeSet<Integer> constructingSubCluster = new TreeSet<>();
                    LinkedList<Integer> unexploredSubClusterPoints = new LinkedList<>();
                    boolean anyEdges = false;
                    boolean incrementedChildCount = false;

                    int rootVertex = examinedVertices.last();
                    constructingSubCluster.add(rootVertex);
                    unexploredSubClusterPoints.add(rootVertex);
                    examinedVertices.remove(rootVertex);

                    // Explore this potential child cluster as long as there are unexplored points:
                    while (!unexploredSubClusterPoints.isEmpty()) {
                        int vertexToExplore = unexploredSubClusterPoints.poll();

                        for (int neighbor : emst.getEdgeListForVertex(vertexToExplore)) {
                            anyEdges = true;
                            if (constructingSubCluster.add(neighbor)) {
                                unexploredSubClusterPoints.add(neighbor);
                                examinedVertices.remove(neighbor);
                            }
                        }

                        // Check if this potential child cluster is a valid cluster:
                        if (!incrementedChildCount && constructingSubCluster.size() >= minClusterSize && anyEdges) {
                            incrementedChildCount = true;
                            numChildClusters++;

                            //If this is the first valid child cluster, stop exploring it:
                            if (firstChildCluster == null) {
                                firstChildCluster = constructingSubCluster;
                                unexploredFirstChildClusterPoints = unexploredSubClusterPoints;
                                break;
                            }
                        }
                    }

                    //If there could be a split, and this child cluster is valid:
                    if (numChildClusters >= 2 && constructingSubCluster.size() >= minClusterSize && anyEdges) {

                        //Check this child cluster is not equal to the unexplored first child cluster:
                        int firstChildClusterMember = firstChildCluster.last();
                        if (constructingSubCluster.contains(firstChildClusterMember))
                            numChildClusters--;

                            //Otherwise, create a new cluster:
                        else {
                            HdbscanCluster parentCluster = clusters.get(examinedClusterLabel);
                            HdbscanCluster newCluster = parentCluster.createNewCluster(constructingSubCluster, currentClusterLabels,
                                                                                       nextClusterLabel, currentEdgeWeight);
                            newClusters.add(newCluster);
                            clusters.add(newCluster);
                            nextClusterLabel++;
                        }
                    }

                    // If this child cluster is not valid cluster, assign it to noise:
                    else if (constructingSubCluster.size() < minClusterSize || !anyEdges){
                        HdbscanCluster parentCluster = clusters.get(examinedClusterLabel);
                        parentCluster.createNewCluster(constructingSubCluster, currentClusterLabels, OUTLIER_NOISE_CLUSTER_LABEL, currentEdgeWeight);

                        for (int point : constructingSubCluster) {
                            pointNoiseLevels[point] = currentEdgeWeight;
                            pointLastClusters[point] = examinedClusterLabel;
                        }
                    }
                }

                //Finish exploring and cluster the first child cluster if there was a split and it was not already clustered:
                if (numChildClusters >= 2 && currentClusterLabels[firstChildCluster.first()] == examinedClusterLabel) {

                    while (!unexploredFirstChildClusterPoints.isEmpty()) {
                        int vertexToExplore = unexploredFirstChildClusterPoints.poll();

                        for (int neighbor : emst.getEdgeListForVertex(vertexToExplore)) {
                            if (firstChildCluster.add(neighbor))
                                unexploredFirstChildClusterPoints.add(neighbor);
                        }
                    }

                    HdbscanCluster parentCluster = clusters.get(examinedClusterLabel);
                    HdbscanCluster newCluster = parentCluster.createNewCluster(firstChildCluster, currentClusterLabels,
                                                                               nextClusterLabel, currentEdgeWeight);
                    newClusters.add(newCluster);
                    clusters.add(newCluster);
                    nextClusterLabel++;
                }
            }

            // increment the current level of the hierarchy:
            if (nextLevelSignificant || !newClusters.isEmpty()) {
                // this is used for below to set the hierarchy level
                lineCount++;
            }

            for (HdbscanCluster newCluster : newClusters) {
                int[] hierarchyLevelLabels = new int[emst.getNumVertices()];
                System.arraycopy(currentClusterLabels, 0, hierarchyLevelLabels, 0, currentClusterLabels.length);
                newCluster.setHierarchyLevel(lineCount);
                hierarchy.put(lineCount, hierarchyLevelLabels);
            }

            if (newClusters.isEmpty())
                nextLevelSignificant = false;
            else
                nextLevelSignificant = true;
        }
        return clusters;
    }

    /**
     * Propagates stability and lowest child death level from each child cluster to its parent cluster in the tree.
     * This step is required before calling prominent clusters can be identified or outlier scores can be calculated.
     * @param clusters A list of clusters representing the cluster tree.
     */
    private static void propagateTree(List<HdbscanCluster> clusters) {
        TreeMap<Integer, HdbscanCluster> clustersToExamine = new TreeMap<>();
        BitSet addedToExaminationList = new BitSet(clusters.size());

        // Find all leaf clusters in the cluster tree:
        for (HdbscanCluster cluster : clusters) {
            if (cluster != null && !cluster.hasChildren()) {
                clustersToExamine.put(cluster.getLabel(), cluster);
                addedToExaminationList.set(cluster.getLabel());
            }
        }

        // Iterate through every cluster, propagating stability from children to parents:
        while (!clustersToExamine.isEmpty()) {
            HdbscanCluster currentCluster = clustersToExamine.pollLastEntry().getValue();
            currentCluster.propagate();

            if (currentCluster.getParent() != null) {
                HdbscanCluster parent = currentCluster.getParent();

                if (!addedToExaminationList.get(parent.getLabel())) {
                    clustersToExamine.put(parent.getLabel(), parent);
                    addedToExaminationList.set(parent.getLabel());
                }
            }
        }
    }

    /**
     * Produces the cluster labels using cluster stability values.
     *
     * @param hierarchy The HDBSCAN* hierarchy.
     * @param clusters A list of clusters representing the cluster tree.
     * @param numPoints The number of points in the dataset.
     * @return A list containing every point's cluster label.
     */
    private static List<Integer> findProminentClusters(Map<Integer, int[]> hierarchy, List<HdbscanCluster> clusters,
                                                      int numPoints) {

        // Take the list of propagated clusters from the root cluster:
        List<HdbscanCluster> solution = clusters.get(1).getPropagatedDescendants();

        List<Integer> clusterLabels = new ArrayList<>(Collections.nCopies(numPoints, OUTLIER_NOISE_CLUSTER_LABEL));

        // Store all the levels at which to find the birth points for the flat clustering:
        TreeMap<Integer, List<Integer>> significantLevels = new TreeMap<>();
        for (HdbscanCluster cluster: solution) {
            List<Integer> clusterList = significantLevels.get(cluster.getHierarchyLevel());

            if (clusterList == null) {
                clusterList = new ArrayList<>();
                significantLevels.put(cluster.getHierarchyLevel(), clusterList);
            }
            clusterList.add(cluster.getLabel());
        }

        //Go through the hierarchy file, setting labels for the flat clustering:
        while (!significantLevels.isEmpty()) {
            Map.Entry<Integer, List<Integer>> entry = significantLevels.pollFirstEntry();
            List<Integer> clusterList = entry.getValue();
            Integer hierarchyLevel = entry.getKey();

            int[] hierarchyLevelLabels = hierarchy.get(hierarchyLevel);

            for (int i = 0; i < hierarchyLevelLabels.length; i++) {
                int label = hierarchyLevelLabels[i];
                if (clusterList.contains(label))
                    clusterLabels.set(i, label);
            }
        }
        return clusterLabels;
    }

    /**
     * Produces the outlier score for each point in the data.
     *
     * @param clusters A list of Clusters forming a cluster tree which has already been propagated.
     * @param pointNoiseLevels A double[] with the levels at which each point becomes noise.
     * @param pointLastClusters An int[] with the last label of each point before becoming noise.
     * @return A {@link DenseVector} containing the outlier scores.
     */
    private static DenseVector calculateOutlierScores(double[] pointNoiseLevels, int[] pointLastClusters,
                                                                 List<HdbscanCluster> clusters) {

        int numPoints = pointNoiseLevels.length;
        DenseVector outlierScores = new DenseVector(numPoints);

        // Iterate through each point, calculating its outlier score:
        for (int i = 0; i < numPoints; i++) {
            double epsilon_max = clusters.get(pointLastClusters[i]).getPropagatedLowestChildDeathLevel();
            double epsilon = pointNoiseLevels[i];

            double score = 0;
            if (epsilon != 0)
                score = 1-(epsilon_max/epsilon);

            outlierScores.set(i, score);
        }
        return outlierScores;
    }

    /**
     * Generate a map keyed by the cluster label and the data points assigned to them. The representation of a data
     * points are their index in the dataset, and their outlier score. In this case, the data points are being sorted
     * by their outlier scores, as convenience to quickly establish the exemplars from the dataset.
     *
     * @param clusterLabels A list containing every point's cluster label.
     * @param outlierScoresVector A {@link DenseVector} containing the outlier scores.
     * @return A map of the cluster labels, and the points assigned to them.
     */
    private static Map<Integer, TreeMap<Double, Integer>> generateClusterAssignments(List<Integer> clusterLabels,
                                                                           DenseVector outlierScoresVector) {
        Map<Integer, TreeMap<Double, Integer>> clusterAssignments = new HashMap<>();
        for (int i = 0; i < clusterLabels.size(); i++) {
            Integer clusterLabel = clusterLabels.get(i);
            Double outlierScore = outlierScoresVector.get(i);
            if (clusterAssignments.containsKey(clusterLabel)) {
                clusterAssignments.get(clusterLabel).put(outlierScore, i);
            }
            else {
                TreeMap<Double, Integer> outlierScoreIndexTree = new TreeMap<>();
                outlierScoreIndexTree.put(outlierScore, i);
                clusterAssignments.put(clusterLabel, outlierScoreIndexTree);
            }
        }
        return clusterAssignments;
    }

    /**
     * Compute the exemplars. These are representative points which are subsets of their clusters, and will be used for
     * prediction on unseen data points.
     *
     * @param data An array of {@link DenseVector} containing the data.
     * @param clusterAssignments A map of the cluster labels, and the points assigned to them.
     * @return A list of {@link ClusterExemplar}s which are used for predictions.
     */
    private static List<ClusterExemplar> computeExemplars(SGDVector[] data, Map<Integer, TreeMap<Double, Integer>> clusterAssignments) {
        List<ClusterExemplar> clusterExemplars = new ArrayList<>();
        int numExemplarsFractionalSize = (int) Math.round(data.length * NUM_EXEMPLARS_FRACTIONAL_VALUE);
        int mumExemplars =  Math.min(numExemplarsFractionalSize, MAX_NUM_EXEMPLARS);

        for (Entry<Integer, TreeMap<Double, Integer>> e : clusterAssignments.entrySet()) {
            int clusterLabel = e.getKey();
            TreeMap<Double, Integer> outlierScoreIndexTree = clusterAssignments.get(clusterLabel);
            int numExemplarsThisCluster = e.getValue().size() * mumExemplars / data.length;
            if (clusterLabel != OUTLIER_NOISE_CLUSTER_LABEL) {
                for (int i = 0; i < numExemplarsThisCluster; i++) {
                    // Note that for non-outliers, the first node is polled from the tree, which has the lowest outlier
                    // score out of the remaining points assigned this cluster.
                    Integer dataIndex = outlierScoreIndexTree.pollFirstEntry().getValue();
                    clusterExemplars.add(new ClusterExemplar(clusterLabel, data[dataIndex]));
                }
            }
            else {
                for (int i = 0; i < numExemplarsThisCluster; i++) {
                    // Note that for outliers the last node is polled from the tree, which has the highest outlier score
                    // out of the remaining points assigned this cluster.
                    Integer dataIndex = outlierScoreIndexTree.pollLastEntry().getValue();
                    clusterExemplars.add(new ClusterExemplar(clusterLabel, data[dataIndex]));
                }
            }
        }
        return clusterExemplars;
    }

    /**
     * Calculates the distance between two vectors.
     *
     * @param vector1 A {@link SGDVector} representing a data point.
     * @param vector2 A {@link SGDVector} representing a second data point.
     * @param distanceType The distance metric to employ.
     * @return A double representing the distance between the two points.
     */
    private static double getDistance(SGDVector vector1, SGDVector vector2, Distance distanceType) {
        double distance;
        switch (distanceType) {
            case EUCLIDEAN:
                distance = vector1.euclideanDistance(vector2);
                break;
            case COSINE:
                distance = vector1.cosineDistance(vector2);
                break;
            case L1:
                distance = vector1.l1Distance(vector2);
                break;
            default:
                throw new IllegalStateException("Unknown distance " + distanceType);
        }
        return distance;
    }

    @Override
    public String toString() {
        return "KMeansTrainer(centroids=" + minClusterSize + ",distanceType=" + distanceType + ",k=" + k + ",numThreads=" + numThreads + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    // TODO: Does this need some basic comments?
    static class ClusterExemplar extends Pair<Integer, SGDVector> {
        ClusterExemplar(Integer a, SGDVector b) {
            super(a, b);
        }
    }

}