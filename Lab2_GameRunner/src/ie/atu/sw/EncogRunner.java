package ie.atu.sw;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class EncogRunner {
    public static void main(String[] args) {
        // Normalize "Number of Enemies" to [0,1] range
        double[][] data = {
                { 2, 0, 0, 0.0 }, { 2, 0, 0, 0.5 }, { 2, 0, 1, 0.5 }, { 2, 0, 1, 1.0 }, { 2, 1, 0, 1.0 },
                { 2, 1, 0, 0.5 }, { 1, 0, 0, 0.0 }, { 1, 0, 0, 0.5 }, { 1, 0, 1, 0.5 }, { 1, 0, 1, 1.0 },
                { 1, 1, 0, 1.0 }, { 1, 1, 0, 0.5 }, { 0, 0, 0, 0.0 }, { 0, 0, 0, 0.5 }, { 0, 0, 1, 0.5 },
                { 0, 0, 1, 1.0 }, { 0, 1, 0, 1.0 }, { 0, 1, 0, 0.5 }
        };

        double[][] expected = {
                { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 },
                { 0.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 },
                { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 },
                { 0.0, 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 0.0 },
                { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 1.0 }
        };

        // Create dataset
        MLDataSet trainingSet = new BasicMLDataSet(data, expected);

        // Define neural network
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 4));  // Input Layer
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5)); // Hidden Layer
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 4)); // Output Layer
        network.getStructure().finalizeStructure();
        network.reset();

        // Train the network
        ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        double minError = 0.09;
        int epoch = 1;
        do {
            train.iteration();
            epoch++;
        } while (train.getError() > minError);
        train.finishTraining();

        // Test the trained neural network with the training data
        for (MLDataPair pair : trainingSet) { 
            
            // Compute the network's output for the given input data
            MLData output = network.compute(pair.getInput());

            // Print the input values and the corresponding network output
            System.out.println(
            "input " + pair.getInput().getData(0) + ","  // First input feature (e.g., Health)
                + pair.getInput().getData(1)      // Second input feature (e.g., Has a Sword)
                + ", Y=" + (int) Math.round(output.getData(0))  // Rounded predicted output (Y)
                + ", Yd=" + (int) pair.getIdeal().getData(0)    // Expected output (Yd)
            ); 
        }


    }

    // Helper function to round output values
    private static String roundArray(double[] data) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < data.length; i++) {
            sb.append((int) Math.round(data[i]));
            if (i < data.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}