package org.tribuo.neighbours;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.example.GaussianClusterDataSource;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQuery;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForce;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.math.neighbour.kdtree.KDTreeFactory;
import org.tribuo.util.Util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * TODO: Delete this right before creating PR, maybe make a branch first. Delete the .csv files here too.
 * This is a test for testing the performance of the NeighbourQuery implementations.
 *
 */
public class TestBreakdownKDTreePerf {

    private static final Logger logger = Logger.getLogger(NeighboursBruteForce.class.getName());

    @BeforeAll
    public static void setup() {
        logger.setLevel(Level.INFO);
    }

    private static SGDVector[] get2DVectors(int numSamples, long seed) {
        DataSource<ClusterID> dataSource = new GaussianClusterDataSource(numSamples, seed);
        Dataset<ClusterID> dataset = new MutableDataset<>(dataSource);

        ImmutableFeatureMap featureMap = dataset.getFeatureIDMap();
        SGDVector[] vectors = new SGDVector[dataset.size()];
        int n = 0;
        for (Example<ClusterID> example : dataset) {
            vectors[n] = DenseVector.createDenseVector(example, featureMap, false);
            n++;
        }
        return vectors;
    }

    private static SGDVector[] getSGDVectorsFromCSV(String filePath, boolean fileContainsHeader) {
        List<SGDVector> vectorList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;

            if (fileContainsHeader) {
                line = br.readLine();
                System.out.println("The header line: " + line + " was ignored.");
            }

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] doubles = new double[values.length];
                for (int i=0; i<values.length; i++) {
                    doubles[i] = Double.parseDouble(values[i]);
                }
                SGDVector vector = DenseVector.createDenseVector(doubles);
                vectorList.add(vector);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("\nThere is a problem with the file " + filePath);
        }
        return vectorList.toArray(new SGDVector[0]);
    }

    private static void constructNQandExecuteQuery(NeighboursQueryFactory nqf, SGDVector[] data, int k) {
        long startTime = System.currentTimeMillis();
        NeighboursQuery nq = nqf.createNeighboursQuery(data);
        long endTime = System.currentTimeMillis();
        logger.info("Construction took " + Util.formatDuration(startTime,endTime));
        nq.queryAll(k);
        endTime = System.currentTimeMillis();
        logger.info("Query took " + Util.formatDuration(startTime,endTime));
        logger.info("");
    }

    private static void doTestIteration(KDTreeFactory kdtFactory) {
        SGDVector[] data = get2DVectors(50000, 1L);

        logger.info("KDTree: small dataset, 2 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        data = get2DVectors(100000, 1L);

        logger.info("KDTree: medium dataset, 2 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        data = get2DVectors(200000, 1L);

        logger.info("KDTree: large dataset, 2 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        logger.info("KDTree: large dataset, 2 features, large k");
        constructNQandExecuteQuery(kdtFactory, data, 20);
        logger.info("");

        String filename = "gaussians-10features.csv";
        String filepath = TestBreakdownKDTreePerf.class.getClassLoader().getResource(filename).getPath();
        data = getSGDVectorsFromCSV(filepath, true);

        logger.info("KDTree: medium dataset, 10 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        logger.info("KDTree: medium dataset, 10 features, medium k");
        constructNQandExecuteQuery(kdtFactory, data, 10);
        logger.info("");

        filename = "gaussians-20features.csv";
        filepath = TestBreakdownKDTreePerf.class.getClassLoader().getResource(filename).getPath();
        data = getSGDVectorsFromCSV(filepath, true);

        logger.info("KDTree: medium dataset, 20 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        filename = "integers-250K-4features.csv";
        filepath = TestBreakdownKDTreePerf.class.getClassLoader().getResource(filename).getPath();
        data = getSGDVectorsFromCSV(filepath, true);

        logger.info("KDTree: large integer dataset, 8 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");
    }

    private static void doAdversarialTestIteration(KDTreeFactory kdtFactory) {
        String filename = "adversarial-integers.csv";
        String filepath = TestBreakdownKDTreePerf.class.getClassLoader().getResource(filename).getPath();
        SGDVector[] data = getSGDVectorsFromCSV(filepath, true);

        logger.info("Adversarial Integer query test - only for debug purposes.");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");
    }

    @Disabled
    @Test
    public void testMultiThreadQueries() {
        KDTreeFactory kdtFactory = new KDTreeFactory(DistanceType.L2, 4);

        logger.info("PERFORMING MULTI-THREADED TESTS...");
        logger.info("");
        doTestIteration(kdtFactory);
    }

    @Test
    public void testAdversarial() {
        KDTreeFactory kdtFactory = new KDTreeFactory(DistanceType.L2, 4);

        logger.info("PERFORMING Adversarial test...");
        logger.info("");
        doAdversarialTestIteration(kdtFactory);
    }
}
