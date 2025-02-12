import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

import java.io.File;

public class IrisClassifier {
    private static final String FILE_PATH = "C:\\Users\\david\\Desktop\\d\\Y4_ARTIFICIAL_INTELLIGENCE_LABS\\K\\src\\iris.txt";

    public void go(String file, String modelType) {
        // Load dataset
        VersatileMLDataSet data = loadData(file);

        // Create Model (MLP or RBF)
        EncogModel model = new EncogModel(data);
        model.selectMethod(data, modelType); 
        data.normalize();

        // Train the Model
        model.holdBackValidation(0.3, true, 1001);
        model.selectTrainingType(data);
        MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);

        // Display Training & Validation Error
        System.out.println("Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
        System.out.println("Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));

        // Test the Model
        testModel(file, bestMethod, data);

        // Shutdown Encog
        Encog.getInstance().shutdown();
    }

    private VersatileMLDataSet loadData(String file) {
        CSVDataSource source = new CSVDataSource(new File(file), false, CSVFormat.DECIMAL_POINT);
        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
        data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
        data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
        data.defineSourceColumn("petal-width", 3, ColumnType.continuous);
        ColumnDefinition out = data.defineSourceColumn("species", 4, ColumnType.nominal);
        data.analyze();
        data.defineSingleOutputOthersInput(out);
        return data;
    }

    private void testModel(String file, MLRegression bestMethod, VersatileMLDataSet data) {
        System.out.println("Testing Model...");
        ReadCSV csv = new ReadCSV(new File(file), false, CSVFormat.DECIMAL_POINT);
        String[] line = new String[4];
        MLData input = data.getNormHelper().allocateInputVector();
        
        while (csv.next()) {
            line[0] = csv.get(0);
            line[1] = csv.get(1);
            line[2] = csv.get(2);
            line[3] = csv.get(3);
            String expected = csv.get(4);
            
            data.getNormHelper().normalizeInputVector(line, input.getData(), false);
            MLData output = bestMethod.compute(input);
            String actual = data.getNormHelper().denormalizeOutputVectorToString(output)[0];

            System.out.println("Expected: " + expected + " Actual: " + actual);
        }
    }

    public static void main(String[] args) {
        IrisClassifier classifier = new IrisClassifier();
        
        // Run MLP model
        System.out.println("\n--- Running Feed-Forward Neural Network (MLP) ---");
        classifier.go(FILE_PATH, MLMethodFactory.TYPE_FEEDFORWARD);

        // Run RBF model
        System.out.println("\n--- Running Radial Basis Function Network (RBF) ---");
        classifier.go(FILE_PATH, MLMethodFactory.TYPE_RBFNETWORK);
    }
}
