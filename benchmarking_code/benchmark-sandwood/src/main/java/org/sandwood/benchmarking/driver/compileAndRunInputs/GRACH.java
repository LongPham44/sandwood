package org.sandwood.benchmarking.driver.compileAndRunInputs;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.sandwood.benchmarking.driver.SandwoodBenchmarkDriver.TestData;
import org.sandwood.benchmarking.driver.SandwoodBenchmarkDriver.TestType;

public class GRACH {
    private static final Path RESOURCES_DIR = Path.of("src", "main", "resources");
    private static final Path OBSERVED_DATA_FILE = RESOURCES_DIR.resolve(
            Path.of("inputs", "org", "sandwood", "benchmarking", "observedData", "GARCH", "observed-data.json"));

    public static Map<TestType, TestData> getInputs() {
        ObservedData observedData = readObservedData();

        Map<TestType, TestData> m = new HashMap<>();
        {
            TestData t = new TestData();
            t.inputs.put("yObserved", requireArray(observedData.y, "y"));
            t.inputs.put("sigma1", requireDouble(observedData.sigma1, "sigma1"));
            t.args = new String[] { "yObserved", "sigma1" };
            t.outputNames = new String[] { "mu", "alpha0", "alpha1", "beta1" };
            m.put(TestType.Gibbs, t);
        }
        return m;
    }

    private static ObservedData readObservedData() {
        try(InputStream in = openObservedData()) {
            ObjectMapper mapper = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
                    false);
            ObservedData observedData = mapper.readValue(in, ObservedData.class);
            requireArray(observedData.y, "y");
            requireDouble(observedData.sigma1, "sigma1");
            validateLength(observedData);
            return observedData;
        } catch(IOException e) {
            throw new IllegalStateException("Failed to read observed data from " + observedDataResource(), e);
        }
    }

    private static InputStream openObservedData() throws IOException {
        String resourceName = observedDataResource();
        InputStream in = GRACH.class.getClassLoader().getResourceAsStream(resourceName);
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

    private static void validateLength(ObservedData observedData) {
        if(observedData.T != null && observedData.T != observedData.y.length) {
            throw new IllegalStateException("Observed data length mismatch in " + observedDataResource()
                    + ": T is " + observedData.T + " but y has length " + observedData.y.length);
        }
    }

    private static double[] requireArray(double[] value, String name) {
        if(value == null) {
            throw new IllegalStateException(
                    "Missing observed data array \"" + name + "\" in " + observedDataResource());
        }
        return value;
    }

    private static double requireDouble(Double value, String name) {
        if(value == null) {
            throw new IllegalStateException(
                    "Missing observed data double \"" + name + "\" in " + observedDataResource());
        }
        return value;
    }

    private static class ObservedData {
        public Integer T;
        public double[] y;
        public Double sigma1;
    }
}