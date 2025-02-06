import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.strategy.Strategy;
import org.encog.ml.train.strategy.end.EndIterationsStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class ZooNeuralNetwork {
    
    // Constants
    private static final int INPUT_NEURONS = 16; // 16 Features from Zoo dataset
    private static final int HIDDEN_NEURONS = 10; // Adjustable hidden layer size
    private static final int OUTPUT_NEURONS = 7; // 7 different classes of animals
    
    public static void main(String[] args) {
        // Load the dataset
        double[][] inputData = new double[101][INPUT_NEURONS];
        double[][] idealOutput = new double[101][OUTPUT_NEURONS];
        loadZooDataset("C:\\Users\\user\\Desktop\\AI\\ARTIFICIAL_INTELLIGENCE_LABS\\Lab2_GameRunner\\src\\zoo.data", inputData, idealOutput);

        // Create training dataset
        BasicMLDataSet trainingSet = new BasicMLDataSet(inputData, idealOutput);

        // Build Neural Network
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, INPUT_NEURONS)); // Input layer
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, HIDDEN_NEURONS)); // Hidden layer
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, OUTPUT_NEURONS)); // Output layer
        network.getStructure().finalizeStructure();
        network.reset();

        // Train the Network
        ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        train.addStrategy(new EndIterationsStrategy(5000)); // Stop training after 5000 iterations if needed
        
        double minError = 0.5; // Lower error for better accuracy
        int epoch = 1;
        
        System.out.println("Training started...");
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error: " + train.getError());
            epoch++;
        } while (train.getError() > minError);
        train.finishTraining();
        System.out.println("Training complete!");

        // Test the Network
        System.out.println("Testing the network...");
        for (MLDataPair pair : trainingSet) {
            MLData output = network.compute(pair.getInput());

            int predictedClass = getMaxIndex(output.getData()); // Convert output to class
            int actualClass = getMaxIndex(pair.getIdeal().getData());

            System.out.println("Input: " + Arrays.toString(pair.getInput().getData()) +
                    ", Predicted: " + predictedClass + ", Actual: " + actualClass);
        }
    }

    // Function to load and preprocess the Zoo dataset
    private static void loadZooDataset(String fileName, double[][] inputData, double[][] idealOutput) {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            int rowIndex = 0;
            
            while ((line = br.readLine()) != null && rowIndex < 101) {
                String[] values = line.split(",");
                
                // Extract input features (first 16 values)
                for (int i = 1; i <= INPUT_NEURONS; i++) {
                    inputData[rowIndex][i - 1] = Double.parseDouble(values[i]);
                }

                // One-hot encoding for the class label (last value in dataset)
                int classIndex = Integer.parseInt(values[17]) - 1; // Convert class to zero-based index
                idealOutput[rowIndex][classIndex] = 1.0;

                rowIndex++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Function to get the index of the maximum value in an array
    private static int getMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex + 1; // Convert back to 1-based class labels
    }
}
