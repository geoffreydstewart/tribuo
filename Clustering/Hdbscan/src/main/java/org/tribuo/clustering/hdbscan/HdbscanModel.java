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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.*;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.hdbscan.HdbscanTrainer.Distance;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

import java.util.*;

/**
 * An HDBSCAN* model. Update all the rest of this!
 * <p>
 * The predict method of this model assigns centres to the provided input,
 * but it does not update the model's centroids.
 * <p>
 * The predict method is single threaded.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public class HdbscanModel extends Model<ClusterID> {
    private static final long serialVersionUID = 1L;

    private final List<Integer> clusterLabels;

    private final DenseVector outlierScoresVector;

    private final Distance distanceType;

    private final List<HdbscanTrainer.ClusterExemplar> clusterExemplars;

    HdbscanModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap,
                 ImmutableOutputInfo<ClusterID> outputIDInfo, List<Integer> clusterLabels, DenseVector outlierScoresVector,
                 List<HdbscanTrainer.ClusterExemplar> clusterExemplars, Distance distanceType) {
        super(name,description,featureIDMap,outputIDInfo,true);
        this.clusterLabels = clusterLabels;
        this.outlierScoresVector = outlierScoresVector;
        this.clusterExemplars = clusterExemplars;
        this.distanceType = distanceType;
    }

    /**
     * Returns the cluster labels.
     * <p>
     * This seems pretty standard, but maybe write more stuff.
     * @return The cluster labels for all the data points.
     */
    public List<Integer> getClusterLabels() {
        return new ArrayList<>(clusterLabels);
    }

    /**
     * Returns the outlier scores.
     * <p>
     * This seems pretty standard, but maybe write more stuff.
     * @return The outlier scores for all the data points.
     */
    public DenseVector getOutlierScoresVector() {
        return outlierScoresVector.copy();
    }

    @Override
    public Prediction<ClusterID> predict(Example<ClusterID> example) {
        SparseVector vector = SparseVector.createSparseVector(example,featureIDMap,false);
        if (vector.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example);
        }
        double minDistance = Double.POSITIVE_INFINITY;
        int clusterLabel = -1;
        for (HdbscanTrainer.ClusterExemplar clusterExemplar : clusterExemplars) {
            double distance;
            switch (distanceType) {
                case EUCLIDEAN:
                    distance = clusterExemplar.getB().euclideanDistance(vector);
                    break;
                case COSINE:
                    distance = clusterExemplar.getB().cosineDistance(vector);
                    break;
                case L1:
                    distance = clusterExemplar.getB().l1Distance(vector);
                    break;
                default:
                    throw new IllegalStateException("Unknown distance " + distanceType);
            }
            if (distance < minDistance) {
                minDistance = distance;
                clusterLabel = clusterExemplar.getA();
            }
        }
        return new Prediction<>(new ClusterID(clusterLabel),vector.size(),example);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<ClusterID>> getExcuse(Example<ClusterID> example) {
        return Optional.empty();
    }

    @Override
    protected HdbscanModel copy(String newName, ModelProvenance newProvenance) {
        DenseVector copyOutlierScoresVector = outlierScoresVector.copy();
        List<Integer> copyClusterLabels = new ArrayList<>(clusterLabels);
        List<HdbscanTrainer.ClusterExemplar> copyExemplars = new ArrayList<>(clusterExemplars);
        return new HdbscanModel(newName, newProvenance, featureIDMap, outputIDInfo, copyClusterLabels,
            copyOutlierScoresVector, copyExemplars, distanceType);
    }
}
