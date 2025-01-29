
public class Runner {
    public static void main(String[] args) {
        // Define the training data: 4 input pairs for logical operations (AND/OR).
        float[][] data = {
            {0.0f, 0.0f}, // Input 1: (0, 0)
            {1.0f, 0.0f}, // Input 2: (1, 0)
            {0.0f, 1.0f}, // Input 3: (0, 1)
            {1.0f, 1.0f}  // Input 4: (1, 1)
        };

        // Logical AND operation
        System.out.println("Logical AND:");
        
        // Define the expected output for the AND operation.
        // Only (1, 1) results in 1, all others result in 0.
        float[] expected = {0.0f, 0.0f, 0.0f, 1.0f};

        // Create a perceptron with 2 inputs (for binary operations).
        var p = new Perceptron(2);
        
        System.out.println("Before training AND");
        // Train the perceptron using the training data and expected outputs.
        // Train for a maximum of 10,000 epochs (iterations) to converge.
        p.train(data, expected, 10000);

        // Test the perceptron on each input pair and print the results.
        for (int row = 0; row < data.length; row++) {
            int result = p.activate(data[row]); // Compute the perceptron output.
            System.out.println("Result " + row + ": " + result); // Print result.
        }

        // Logical OR operation
        System.out.println("\nLogical OR:");
        
        // Define the expected output for the OR operation.
        // Any input with at least one "1" results in 1, otherwise 0.
        float[] expected1 = {0.0f, 1.0f, 1.0f, 1.0f};

        // Retrain the perceptron with the new expected outputs for OR.
        p.train(data, expected1, 10000);

        // Test the perceptron on each input pair for the OR operation.
        for (int row = 0; row < data.length; row++) {
            int result = p.activate(data[row]); // Compute the perceptron output.
            System.out.println("Result " + row + ": " + result); // Print result.
        }
    }
}
