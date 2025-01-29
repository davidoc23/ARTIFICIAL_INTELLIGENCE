import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Perceptron {
    private float[] weights;  // Array of weights for the perceptron connections
    private float theta = 0.2f; // Threshold (bias)

    // Constructor initializes weights randomly and prints the perceptron details
    public Perceptron(int connection) {
        weights = new float[connection];  // Initialize weight array with given number of connections
        var rand = ThreadLocalRandom.current();  // Thread-safe random generator
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextFloat(-1.0f, 1.0f);  // Initialize weights with random values between -1 and 1
        }
        System.out.println(this);  // Print the initial state of the perceptron
    }

    // Getter method to access the weights
    public float[] getWeights() {
        return weights;
    }

    // Activation function to calculate the output (0 or 1) based on inputs and weights
    public int activate(float[] inputs) {
        float sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];  // Weighted sum of inputs
        }
        return sum >= theta ? 1 : 0;  // If sum is greater than or equal to the threshold (theta), output 1, else 0
    }

    // Training method to adjust the weights using the perceptron learning rule
    public void train(float[][] data, float[] expected, int max_epochs) {
        float alpha = 0.1f; // Learning rate
        // Loop through each epoch (iteration) of training
        for (int epoch = 0; epoch < max_epochs; epoch++) {
            int errorCount = 0;  // Counter to track the number of errors in this epoch
            // Loop through each data point
            for (int i = 0; i < data.length; i++) {
                int result = activate(data[i]);  // Get the perceptron output for the current input
                float error = expected[i] - result;  // Calculate the error (difference between expected and result)
                if (error != 0) {
                    errorCount++;  // Increment error count if there is a mismatch
                    // Update the weights based on the error using the perceptron learning rule
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += alpha * error * data[i][j];  // Adjust weights
                    }
                }
            }
            // If no errors were encountered, training is complete and we break out of the loop
            if (errorCount == 0) {
                System.out.println("Training complete in " + (epoch + 1) + " epochs.");
                break;
            }
        }
        System.out.println(this);  // Print the final state of the perceptron after training
    }

    // Override toString method to represent the perceptron with its weights
    @Override
    public String toString() {
        return "Perceptron [weights=" + Arrays.toString(weights) + "]";  // Returns a string representation of the perceptron with its weights
    }
}
