package org.tribuo.clustering.hdbscan;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.EmptyResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.util.Util;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class TestPerfHdbscan {

    private static Dataset<ClusterID> get2KDataset() {
        ClusteringFactory clusteringFactory = new ClusteringFactory();
        ResponseProcessor<ClusterID> emptyResponseProcessor = new EmptyResponseProcessor<>(clusteringFactory);
        Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();
        regexMappingProcessors.put("Feature1", new DoubleFieldProcessor("Feature1"));
        regexMappingProcessors.put("Feature2", new DoubleFieldProcessor("Feature2"));
        regexMappingProcessors.put("Feature3", new DoubleFieldProcessor("Feature3"));
        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(emptyResponseProcessor,regexMappingProcessors);
        File file = new File(TestPerfHdbscan.class.getClassLoader().getResource("basic-gaussians.csv").getFile());
        CSVDataSource<ClusterID> csvDataSource = new CSVDataSource<>(file.toPath(), rowProcessor, false);
        return new MutableDataset<>(csvDataSource);
    }

    private static Dataset<ClusterID> get50KDataset() {
        ClusteringFactory clusteringFactory = new ClusteringFactory();
        ResponseProcessor<ClusterID> emptyResponseProcessor = new EmptyResponseProcessor<>(clusteringFactory);
        Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();
        regexMappingProcessors.put("Feature1", new DoubleFieldProcessor("Feature1"));
        regexMappingProcessors.put("Feature2", new DoubleFieldProcessor("Feature2"));
        regexMappingProcessors.put("Feature3", new DoubleFieldProcessor("Feature3"));
        regexMappingProcessors.put("Feature4", new DoubleFieldProcessor("Feature4"));
        regexMappingProcessors.put("Feature5", new DoubleFieldProcessor("Feature5"));
        regexMappingProcessors.put("Feature6", new DoubleFieldProcessor("Feature6"));
        regexMappingProcessors.put("Feature7", new DoubleFieldProcessor("Feature7"));

        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(emptyResponseProcessor, regexMappingProcessors);
        File file = new File(TestPerfHdbscan.class.getClassLoader().getResource("50k-gaussians-7features.csv").getFile());
        CSVDataSource<ClusterID> csvDataSource = new CSVDataSource<>(file.toPath(), rowProcessor, false);
        return new MutableDataset<>(csvDataSource);
    }

    private static Dataset<ClusterID> get100KDataset() {
        ClusteringFactory clusteringFactory = new ClusteringFactory();
        ResponseProcessor<ClusterID> emptyResponseProcessor = new EmptyResponseProcessor<>(clusteringFactory);
        Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();
        regexMappingProcessors.put("Feature1", new DoubleFieldProcessor("Feature1"));
        regexMappingProcessors.put("Feature2", new DoubleFieldProcessor("Feature2"));
        regexMappingProcessors.put("Feature3", new DoubleFieldProcessor("Feature3"));
        regexMappingProcessors.put("Feature4", new DoubleFieldProcessor("Feature4"));
        regexMappingProcessors.put("Feature5", new DoubleFieldProcessor("Feature5"));
        regexMappingProcessors.put("Feature6", new DoubleFieldProcessor("Feature6"));
        regexMappingProcessors.put("Feature7", new DoubleFieldProcessor("Feature7"));

        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(emptyResponseProcessor, regexMappingProcessors);
        File file = new File(TestPerfHdbscan.class.getClassLoader().getResource("100k-gaussians-7features.csv").getFile());
        CSVDataSource<ClusterID> csvDataSource = new CSVDataSource<>(file.toPath(), rowProcessor, false);
        return new MutableDataset<>(csvDataSource);
    }

    private static Dataset<ClusterID> get150KDataset() {
        ClusteringFactory clusteringFactory = new ClusteringFactory();
        ResponseProcessor<ClusterID> emptyResponseProcessor = new EmptyResponseProcessor<>(clusteringFactory);
        Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();
        regexMappingProcessors.put("Feature1", new DoubleFieldProcessor("Feature1"));
        regexMappingProcessors.put("Feature2", new DoubleFieldProcessor("Feature2"));
        regexMappingProcessors.put("Feature3", new DoubleFieldProcessor("Feature3"));
        regexMappingProcessors.put("Feature4", new DoubleFieldProcessor("Feature4"));
        regexMappingProcessors.put("Feature5", new DoubleFieldProcessor("Feature5"));
        regexMappingProcessors.put("Feature6", new DoubleFieldProcessor("Feature6"));
        regexMappingProcessors.put("Feature7", new DoubleFieldProcessor("Feature7"));
        regexMappingProcessors.put("Feature8", new DoubleFieldProcessor("Feature8"));
        regexMappingProcessors.put("Feature9", new DoubleFieldProcessor("Feature9"));
        regexMappingProcessors.put("Feature10", new DoubleFieldProcessor("Feature10"));

        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(emptyResponseProcessor, regexMappingProcessors);
        File file = new File(TestPerfHdbscan.class.getClassLoader().getResource("150k-gaussians-10features.csv").getFile());
        CSVDataSource<ClusterID> csvDataSource = new CSVDataSource<>(file.toPath(), rowProcessor, false);
        return new MutableDataset<>(csvDataSource);
    }

    private static void doTrain(HdbscanTrainer trainer, Dataset<ClusterID> dataset, String trainerMessage) {
        long startTime = System.currentTimeMillis();
        trainer.train(dataset);
        long endTime = System.currentTimeMillis();
        System.out.println(trainerMessage + " training took " + Util.formatDuration(startTime,endTime));
        System.out.println();
    }

    @Disabled
    @Test
    public void hdbscan50000ClusteringTest() {
        /*
        Dataset<ClusterID> dataset2K = get2KDataset();
        HdbscanTrainer bf2ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 2, NeighboursQueryFactoryType.BRUTE_FORCE);
        HdbscanTrainer kd2ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 2, NeighboursQueryFactoryType.KD_TREE);

        doTrain(bf2ktrainer, dataset2K, "Brute Force 2K");
        doTrain(kd2ktrainer, dataset2K, "K-D Tree 2K");
        System.out.println("***************");
        System.out.println();
         */

        Dataset<ClusterID> dataset50k = get50KDataset();
        HdbscanTrainer bf50ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 8, NeighboursQueryFactoryType.BRUTE_FORCE);
        HdbscanTrainer kd50ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 8, NeighboursQueryFactoryType.KD_TREE);

        doTrain(bf50ktrainer, dataset50k, "Brute Force 50K");
        doTrain(kd50ktrainer, dataset50k, "K-D Tree 50K");
        System.out.println("***************");
        System.out.println();

        Dataset<ClusterID> dataset100k = get100KDataset();
        HdbscanTrainer bf100ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 8, NeighboursQueryFactoryType.BRUTE_FORCE);
        HdbscanTrainer kd100ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 8, NeighboursQueryFactoryType.KD_TREE);

        doTrain(bf100ktrainer, dataset100k, "Brute Force 100K");
        doTrain(kd100ktrainer, dataset100k, "K-D Tree 100K");
        System.out.println("***************");
        System.out.println();

        Dataset<ClusterID> dataset150k = get150KDataset();
        HdbscanTrainer bf150ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 8, NeighboursQueryFactoryType.BRUTE_FORCE);
        HdbscanTrainer kd150ktrainer = new HdbscanTrainer(10, DistanceType.L2, 5, 8, NeighboursQueryFactoryType.KD_TREE);

        doTrain(bf150ktrainer, dataset150k, "Brute Force 150K");
        doTrain(kd150ktrainer, dataset150k, "K-D Tree 150K");
        System.out.println("***************");
        System.out.println();
    }

}
