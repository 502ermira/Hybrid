package Project4;

/**
 * This code performs classification on the dataset using a hybrid clustering-classification approach.
 * The dataset is first clustered using the FarthestFirst algorithm, and then a NaiveBayes classifier is trained on the clustered data.
 * The performance of the classifier is evaluated using cross-validation.
 */

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.FarthestFirst;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ClusterClassification {
    
    public static void main(String[] args) throws Exception {
        
        // Load the dataset from the ARFF file
        DataSource source = new DataSource("C:\\Users\\Admin\\Desktop\\fina.arff");
        Instances data = source.getDataSet();
        
        // Split the dataset into training and test sets using holdout validation
        RemovePercentage rp = new RemovePercentage();
        rp.setInputFormat(data);
        rp.setPercentage(70);
        Instances testSet = Filter.useFilter(data, rp);
        rp.setInvertSelection(true);
        Instances trainingSet = Filter.useFilter(data, rp);
        
        // Cluster the training set using the FarthestFirst algorithm
        FarthestFirst ff = new FarthestFirst();
        ff.setNumClusters(100);
        ff.buildClusterer(trainingSet);
        
        // Create a new dataset containing the cluster centroids for each instance in the test set
        Instances clusteredTestSet = new Instances(testSet, testSet.numInstances());
        for (int i = 0; i < testSet.numInstances(); i++) {
            Instance inst = testSet.instance(i);
            int clusterIndex = ff.clusterInstance(inst);
            clusteredTestSet.add(ff.getClusterCentroids().instance(clusterIndex));
        }
        clusteredTestSet.setClassIndex(clusteredTestSet.numAttributes() - 1);
        
        // Train a NaiveBayes classifier on the clustered training set and evaluate its performance on the clustered test set using cross-validation
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(clusteredTestSet);
        
        Evaluation eval = new Evaluation(clusteredTestSet);
        eval.crossValidateModel(nb, clusteredTestSet, 10, new Random(1));
        
        // Output the evaluation results
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}