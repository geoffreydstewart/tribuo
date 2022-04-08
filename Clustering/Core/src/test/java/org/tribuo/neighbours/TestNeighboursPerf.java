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
 */
public class TestNeighboursPerf {

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
        nq.queryAll(k);
        long endTime = System.currentTimeMillis();
        logger.info("Construct and Query took " + Util.formatDuration(startTime,endTime));
        logger.info("");
    }

    private static void doTestIteration(NeighboursBruteForceFactory nbfFactory, KDTreeFactory kdtFactory) {
        SGDVector[] data = get2DVectors(200000, 1L);

        logger.info("NeighboursBruteForce: big dataset, 2 features, small k");
        constructNQandExecuteQuery(nbfFactory, data, 5);

        logger.info("KDTree: big dataset, 2 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        String filename = "gaussians-10features.csv";
        String filepath = TestNeighboursPerf.class.getClassLoader().getResource(filename).getPath();
        data = getSGDVectorsFromCSV(filepath, true);

        logger.info("NeighboursBruteForce: medium dataset, 10 features, small k");
        constructNQandExecuteQuery(nbfFactory, data, 5);

        logger.info("KDTree: medium dataset, 10 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        filename = "gaussians-20features.csv";
        filepath = TestNeighboursPerf.class.getClassLoader().getResource(filename).getPath();
        data = getSGDVectorsFromCSV(filepath, true);

        logger.info("NeighboursBruteForce: medium dataset, 20 features, small k");
        constructNQandExecuteQuery(nbfFactory, data, 5);

        logger.info("KDTree: medium dataset, 20 features, small k");
        constructNQandExecuteQuery(kdtFactory, data, 5);
        logger.info("");

        /*logger.info("Target implementation: big dataset, 2 features, medium k");
        executeQuery(nbf, 100);

        logger.info("New implementation: big dataset, 2 features, medium k");
        executeQuery(kdt, 100);
        logger.info("");*/
    }

    @Disabled
    @Test
    public void testSingleThreadedQueries() {
        NeighboursBruteForceFactory nbfFactory = new NeighboursBruteForceFactory(DistanceType.L2, 1);
        KDTreeFactory nbfnFactory = new KDTreeFactory(DistanceType.L2, 1);

        logger.info("PERFORMING SINGLE THREADED TESTS...");
        logger.info("");
        doTestIteration(nbfFactory, nbfnFactory);
    }

    @Disabled
    @Test
    public void testMultiThreadQueries() {
        NeighboursBruteForceFactory nbfFactory = new NeighboursBruteForceFactory(DistanceType.L2, 4);
        KDTreeFactory kdtFactory = new KDTreeFactory(DistanceType.L2, 4);

        logger.info("PERFORMING MULTI-THREADED TESTS...");
        logger.info("");
        doTestIteration(nbfFactory, kdtFactory);
    }
}
