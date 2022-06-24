/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQuery;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.neighbour.kdtree.KDTree;
import org.tribuo.math.neighbour.kdtree.KDTreeFactory;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

class BoruvkaKDTreeAlgorithm {

    int numPoints;
    int numFeatures;
    BitSet components;
    double[] boundsArr;
    int[] componentOfPointArr;
    int[] componentOfNodeArr;
    int[] candidateNeighbourArr;
    int[] candidatePointArr;
    double[] candidateDistanceArr;
    double[] coreDistanceArr;

    UnionFind unionFind;

    int[] nearestMRDNeighbors;
    int[] otherMRDNeighbors;
    double[] nearestMRDDistances;

    int numEdges = 0;

    BoruvkaKDTreeAlgorithm(SGDVector[] data, int k, KDTreeFactory kdTreeFactory) {
        KDTree coreDistTree = kdTreeFactory.createNeighboursQuery(data);

        numPoints = data.length;
        numFeatures = data[0].size();
        components = new BitSet(data.length);
        boundsArr = new double[data.length];
        componentOfPointArr = new int[data.length];
        componentOfNodeArr = new int[data.length];
        candidatePointArr = new int[data.length];
        candidateNeighbourArr = new int[data.length];
        candidateDistanceArr = new double[data.length];
        coreDistanceArr = new double[data.length];

        UnionFind unionFind = new UnionFind(data.length);

        nearestMRDNeighbors = new int[2*data.length - 1];
        otherMRDNeighbors = new int[2*data.length - 1];
        nearestMRDDistances = new double[2*data.length - 1];

        int numEdges = 0;

        for (int i=0; i<data.length; i++) {
            componentOfPointArr[i] = i;
            componentOfNodeArr[i] = -(i+1);
            candidatePointArr[i] = i;
            candidateNeighbourArr[i] = -1;
            candidateDistanceArr[i] = Double.MAX_VALUE;
        }

        // TODO: change to array maybe
        List<List<Integer>> knnIndiciesList = new ArrayList<>();
        List<List<Pair<Integer, Double>>> indexDistancePairListOfLists = coreDistTree.queryAll(k);
        for (int i=0; i < data.length; i++) {
            List<Integer> knnIndicies = new ArrayList<>();
            // ignore the first index, it's the index of the point itself
            for (int j=1; j < k; j++) {
                knnIndicies.add(indexDistancePairListOfLists.get(i).get(j).getA());
            }
            knnIndiciesList.add(knnIndicies);
            coreDistanceArr[i] = indexDistancePairListOfLists.get(i).get(k-1).getB();
        }

        for (int n=0; n < data.length; n++) {
            for (int j=0; j < k-1; j++) {
                int m = knnIndiciesList.get(n).get(j);
                if (n == m) {
                    continue;
                }
                if (coreDistanceArr[m] <= coreDistanceArr[n]) {
                    candidatePointArr[n] = n;
                    candidateNeighbourArr[n] = m;
                    candidateDistanceArr[n] = coreDistanceArr[n];
                    break;
                }
            }
        }

        updateComponents();

        for (int i=0; i<data.length; i++) {
            boundsArr[i] = Double.MAX_VALUE;
        }
    }

    private void updateComponents() {

        for (int c=0; c < components.size(); c++) {
            if (!components.get(c)) {
                continue;
            }
            int source = candidatePointArr[c];
            int sink = candidateNeighbourArr[c];
            if (source == -1 || sink == -1) {
                continue;
            }
            int currentSourceComponent = unionFind.find(source);
            int currentSinkComponent = unionFind.find(sink);
            if (currentSourceComponent == currentSinkComponent) {
                // these have already been joined
                candidatePointArr[c] = -1;
                candidateNeighbourArr[c] = -1;
                candidateDistanceArr[c] = Double.MAX_VALUE;
                continue;
            }
            nearestMRDNeighbors[numEdges] = source;
            otherMRDNeighbors[numEdges] = sink;
            nearestMRDDistances[numEdges] = candidateDistanceArr[c];
            numEdges++;

            unionFind.union(source, sink);

            // Reset everything,and check if we're done
            candidateDistanceArr[c] = Double.MAX_VALUE;
            if (numEdges == numPoints - 1) {
                components = unionFind.getIsComponent();
                return;
            }
        }

        components = unionFind.getIsComponent();
        for (int i=0; i<numPoints; i++) {
            boundsArr[i] = Double.MAX_VALUE;
        }

    }

    private static final class UnionFind {
        private final int[] parents;
        private final int[] ranks;
        private final BitSet isComponent;

        UnionFind(int size) {
            parents = new int[size];
            ranks = new int[size];
            for (int i = 0; i < size; i++) {
                parents[i] = i;
                ranks[i] = 0;
            }
            isComponent = new BitSet(size);
            isComponent.flip(0, size);
        }

        int find(int n) {
            while (n != parents[n]) {
                n = parents[n];
            }
            return n;
        }

        void union(int x, int y) {
            int xParent = find(x);
            int yParent = find(y);
            if (xParent == yParent) {
                return;
            }

            if (ranks[xParent] < ranks[yParent]) {
                parents[xParent] = yParent;
                isComponent.clear(xParent);
            } else if (ranks[xParent] > ranks[yParent]) {
                parents[yParent] = xParent;
                isComponent.clear(yParent);
            } else {
                parents[yParent] = xParent;
                ranks[xParent]++;
                isComponent.clear(yParent);
            }
        }

        BitSet getIsComponent() {
            return isComponent;
        }
    }
}
