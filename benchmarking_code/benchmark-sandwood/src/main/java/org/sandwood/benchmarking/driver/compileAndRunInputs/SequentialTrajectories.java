package org.sandwood.benchmarking.driver.compileAndRunInputs;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.sandwood.benchmarking.driver.SandwoodBenchmarkDriver.TestData;
import org.sandwood.benchmarking.driver.SandwoodBenchmarkDriver.TestType;

public class SequentialTrajectories {
    private static final Path RESOURCES_DIR = Path.of("src", "main", "resources");
    private static final Path OBSERVED_DATA_FILE = RESOURCES_DIR.resolve(
            Path.of("inputs", "org", "sandwood", "benchmarking", "observedData", "SequentialTrajectories",
                    "observed-data.json"));

    public static Map<TestType, TestData> getInputs() {
        Map<String, double[][]> observedData = readObservedData();

        Map<TestType, TestData> m = new HashMap<>();
        {
            TestData t = new TestData();
            t.inputs.put("obsViews", requireArray(observedData, "obsViews"));
            t.inputs.put("obsViewsLags", requireArray(observedData, "obsViewsLags"));
            t.inputs.put("obsSharesLag1", requireArray(observedData, "obsSharesLag1"));
            t.args = new String[] { "obsViews", "obsViewsLags", "obsSharesLag1" };
            t.outputNames = new String[] { "gamma", "a", "b", "c", "sigma2", "qualities" };
            m.put(TestType.Gibbs, t);
        }
        return m;
    }

    private static Map<String, double[][]> readObservedData() {
        try(InputStream in = openObservedData()) {
            ObjectMapper mapper = new ObjectMapper();
            return mapper.readValue(in, new TypeReference<Map<String, double[][]>>() {});
        } catch(IOException e) {
            throw new IllegalStateException("Failed to read observed data from " + observedDataResource(), e);
        }
    }

    private static InputStream openObservedData() throws IOException {
        String resourceName = observedDataResource();
        InputStream in = SequentialTrajectories.class.getClassLoader().getResourceAsStream(resourceName);
        if(in != null) {
            return in;
        }

        if(Files.isRegularFile(OBSERVED_DATA_FILE)) {
            return Files.newInputStream(OBSERVED_DATA_FILE);
        }

        throw new IOException("Observed data file not found on classpath or filesystem: " + resourceName + " / "
                + OBSERVED_DATA_FILE);
    }

    private static String observedDataResource() {
        return RESOURCES_DIR.relativize(OBSERVED_DATA_FILE).toString().replace('\\', '/');
    }

    private static double[][] requireArray(Map<String, double[][]> observedData, String name) {
        double[][] value = observedData.get(name);
        if(value == null) {
            throw new IllegalStateException(
                    "Missing observed data array \"" + name + "\" in " + observedDataResource());
        }

        // Only take a portion of the two-dimensional array
        int numVideos = 1500;
        return Arrays.copyOfRange(value, 0, numVideos);
    }
}
