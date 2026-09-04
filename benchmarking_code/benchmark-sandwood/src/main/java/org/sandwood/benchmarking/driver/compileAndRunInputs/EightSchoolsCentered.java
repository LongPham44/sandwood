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

public class EightSchoolsCentered {
    private static final Path RESOURCES_DIR = Path.of("src", "main", "resources");
    private static final Path OBSERVED_DATA_FILE = RESOURCES_DIR.resolve(
            Path.of("inputs", "org", "sandwood", "benchmarking", "observedData", "EightSchools",
                    "observed-data.json"));

    public static Map<TestType, TestData> getInputs() {
        ObservedData observedData = readObservedData();

        Map<TestType, TestData> m = new HashMap<>();
        {
            TestData t = new TestData();
            t.inputs.put("yObserved", requireArray(observedData.y, "y"));
            t.inputs.put("sigma", requireArray(observedData.sigma, "sigma"));
            t.args = new String[] { "yObserved", "sigma" };
            t.outputNames = new String[] { "theta", "mu", "tau" };
            m.put(TestType.Gibbs, t);
        }
        return m;
    }

    private static ObservedData readObservedData() {
        try(InputStream in = openObservedData()) {
            ObjectMapper mapper = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
                    false);
            ObservedData observedData = mapper.readValue(in, ObservedData.class);
            requireInteger(observedData.J, "J");
            requireArray(observedData.y, "y");
            requireArray(observedData.sigma, "sigma");
            validateLength(observedData);
            return observedData;
        } catch(IOException e) {
            throw new IllegalStateException("Failed to read observed data from " + observedDataResource(), e);
        }
    }

    private static InputStream openObservedData() throws IOException {
        String resourceName = observedDataResource();
        InputStream in = EightSchoolsCentered.class.getClassLoader().getResourceAsStream(resourceName);
        if(in != null)
            return in;
        if(Files.isRegularFile(OBSERVED_DATA_FILE))
            return Files.newInputStream(OBSERVED_DATA_FILE);
        throw new IOException("Observed data file not found on classpath or filesystem: " + resourceName + " / "
                + OBSERVED_DATA_FILE);
    }

    private static String observedDataResource() {
        return RESOURCES_DIR.relativize(OBSERVED_DATA_FILE).toString().replace('\\', '/');
    }

    private static void validateLength(ObservedData observedData) {
        int j = observedData.J;
        if(observedData.y.length != j || observedData.sigma.length != j)
            throw new IllegalStateException("Observed data length mismatch in " + observedDataResource());
    }

    private static int requireInteger(Integer value, String name) {
        if(value == null)
            throw new IllegalStateException("Missing observed data integer \"" + name + "\" in "
                    + observedDataResource());
        return value;
    }

    private static double[] requireArray(double[] value, String name) {
        if(value == null)
            throw new IllegalStateException("Missing observed data array \"" + name + "\" in "
                    + observedDataResource());
        return value;
    }

    private static class ObservedData {
        public Integer J;
        public double[] y;
        public double[] sigma;
    }
}
