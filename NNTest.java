import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Enmao Diao
 * @version 1.0
 */
public class NNTest {
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static String training_results = "";
    private static String test_results = "";
    private static DecimalFormat df = new DecimalFormat("0.000");

	public static void randomizedHillClimbing(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] train_data, Instance[] test_data) {



		String oaName = "RHC";
		BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, outputlayer});  
	    NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network, measure);
		OptimizationAlgorithm oa = new RandomizedHillClimbing(nno);
        
        // Train the model		
		double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, oaName, iteration, train_data); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();
               
        for(int j = 0; j < train_data.length; j++) {
            network.setInputValues(train_data[j].getData());
            network.run();

            predicted = Double.parseDouble(train_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        training_results =  "\nTest on Training set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(training_results);
	    
        // Test on test set	    
        start = System.nanoTime(); correct = 0; incorrect = 0;
               
        for(int j = 0; j < test_data.length; j++) {
            network.setInputValues(test_data[j].getData());
            network.run();

            predicted = Double.parseDouble(test_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        test_results =  "\nTest on Test set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(test_results);
	}

	public static void simulatedAnnealing(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] train_data, Instance[] test_data, double start_temp, double cooling_rate) {



		String oaName = "SA";
		BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, outputlayer});  
	    NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network, measure);
		OptimizationAlgorithm oa = new SimulatedAnnealing(start_temp, cooling_rate, nno);
        
        // Train the model		
		double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, oaName, iteration, train_data); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();
               
        for(int j = 0; j < train_data.length; j++) {
            network.setInputValues(train_data[j].getData());
            network.run();

            predicted = Double.parseDouble(train_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        training_results =  "\nTest on Training set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(training_results);
	    
        // Test on test set	    
        start = System.nanoTime(); correct = 0; incorrect = 0;
               
        for(int j = 0; j < test_data.length; j++) {
            network.setInputValues(test_data[j].getData());
            network.run();

            predicted = Double.parseDouble(test_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        test_results =  "\nTest on Test set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(test_results);
	}
	
	public static void geneticAlgorithm(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] train_data, Instance[] test_data, int populationSize, int toMate, int toMutate) {



		String oaName = "GA";
		BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, outputlayer});  
	    NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network, measure);
		OptimizationAlgorithm oa = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nno);
        
        // Train the model		
		double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        train(oa, network, oaName, iteration, train_data); //trainer.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();
               
        for(int j = 0; j < train_data.length; j++) {
            network.setInputValues(train_data[j].getData());
            network.run();

            predicted = Double.parseDouble(train_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        training_results =  "\nTest on Training set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(training_results);
	    
        // Test on test set	    
        start = System.nanoTime(); correct = 0; incorrect = 0;
               
        for(int j = 0; j < test_data.length; j++) {
            network.setInputValues(test_data[j].getData());
            network.run();

            predicted = Double.parseDouble(test_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        test_results =  "\nTest on Test set Results for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(test_results);
	}

	public static void backPropagation(int inputlayer, int hiddenlayer, int outputlayer, int iteration, DataSet set, Instance[] train_data, Instance[] test_data) {



		String Name = "BP";
		BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputlayer, hiddenlayer, outputlayer});  
		ConvergenceTrainer trainer = new ConvergenceTrainer(
	               new BatchBackPropagationTrainer(set, network,
	                   new SumOfSquaresError(), new RPROPUpdateRule()));
		trainer.setMaxIterations(iteration);
        // Train the model		
		double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        trainer.train();
        System.out.println("Convergence in " 
            + trainer.getIterations() + " iterations");
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        // Test on training set
        double predicted, actual;
        start = System.nanoTime();
               
        for(int j = 0; j < train_data.length; j++) {
            network.setInputValues(train_data[j].getData());
            network.run();

            predicted = Double.parseDouble(train_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        training_results =  "\nTest on Training set Results for " + Name + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(training_results);
	    
        // Test on test set	    
        start = System.nanoTime(); correct = 0; incorrect = 0;
               
        for(int j = 0; j < test_data.length; j++) {
            network.setInputValues(test_data[j].getData());
            network.run();

            predicted = Double.parseDouble(test_data[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        test_results =  "\nTest on Test set Results for " + Name + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
	    System.out.println(test_results);
	}
	
    private static void learningCurve(Instance[] test_data){
    	int[] splits = new int[]{20,30,40,50,60,70,80};
    	int[] musk_train_lengths = new int[]{923,1385,1847,2309,2770,3232,3694};
    	int musk_train_attributes_length = 166;
    	int inputLayer = 166, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    	int[] iterations = new int[]{100,200,400,600,1000,1500,2000,5000};
    	for(int i=0;i<splits.length;i++){
    		String musk_train_path = "src/musk_training_70_" + splits[i] + "-training.txt";
    		Instance[] train_data = DataParser.getData(musk_train_path,musk_train_lengths[i],musk_train_attributes_length);
    		DataSet train_set = new DataSet(train_data);
    		System.out.println("--------------" + splits[i] + " % training set--------------");
            for(int j=0;j<iterations.length;j++){
	        	System.out.println("--------------" + iterations[j] + " iterations--------------");
		    	randomizedHillClimbing(inputLayer, hiddenLayer, outputLayer, iterations[j], train_set, train_data, test_data);        
		    	simulatedAnnealing(inputLayer, hiddenLayer, outputLayer, iterations[j], train_set, train_data, test_data, 1E11, 0.95);
		    	geneticAlgorithm(inputLayer, hiddenLayer, outputLayer, iterations[j]/10, train_set, train_data, test_data, 50, 20, 10);
		    	backPropagation(inputLayer, hiddenLayer, outputLayer, iterations[j]/10, train_set, train_data, test_data);
            }
    	}
    }
    
    public static void main(String[] args) {
    	String musk_train_path = "clean2_training_70.txt";
    	int musk_train_length = 4618;
    	int musk_train_attributes_length = 166;
    	String musk_test_path = "clean2_test_70.txt";
    	int musk_test_length = 1980;
    	int musk_test_attributes_length = 166;
        Instance[] train_data = DataParser.getData(musk_train_path,musk_train_length,musk_train_attributes_length);
        Instance[] test_data = DataParser.getData(musk_test_path,musk_test_length,musk_test_attributes_length);
        DataSet train_set = new DataSet(train_data);

        int inputLayer = 166, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;	
        int[] iterations = new int[]{100,200,400,600,1000,1500,2000,5000};
        int[] hiddenLayers = new int[]{3,5,10,20,40};
        for(int i=0;i<iterations.length;i++){
        	System.out.println("--------------" + iterations[i] + " iterations--------------");
	    	randomizedHillClimbing(inputLayer, hiddenLayer, outputLayer, iterations[i], train_set, train_data, test_data);        
	    	simulatedAnnealing(inputLayer, hiddenLayer, outputLayer, iterations[i], train_set, train_data, test_data, 1E11, 0.95);
	    	geneticAlgorithm(inputLayer, hiddenLayer, outputLayer, iterations[i]/10, train_set, train_data, test_data, 50, 20, 10);
        	backPropagation(inputLayer, hiddenLayer, outputLayer, iterations[i], train_set, train_data, test_data);
        }
        for(int i=0;i<hiddenLayers.length;i++){
        	System.out.println("--------------" + hiddenLayers[i] + " hidden layers--------------");
            for(int j=0;j<iterations.length;j++){
	        	System.out.println("--------------" + iterations[j] + " iterations--------------");
		    	randomizedHillClimbing(inputLayer, hiddenLayers[i], outputLayer, iterations[j], train_set, train_data, test_data);        
		    	simulatedAnnealing(inputLayer, hiddenLayers[i], outputLayer, iterations[j], train_set, train_data, test_data, 1E11, 0.95);
		    	geneticAlgorithm(inputLayer, hiddenLayers[i], outputLayer, iterations[j]/10, train_set, train_data, test_data, 50, 20, 10);
		    	backPropagation(inputLayer, hiddenLayers[i], outputLayer, iterations[j], train_set, train_data, test_data);
            }
        }
        learningCurve(test_data);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration, Instance[] data) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < iteration; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < data.length; j++) {
                network.setInputValues(data[j].getData());
                network.run();

                Instance output = data[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }
}
