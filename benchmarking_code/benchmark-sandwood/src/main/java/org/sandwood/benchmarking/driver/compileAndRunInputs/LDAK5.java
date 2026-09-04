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

public class LDAK5 {
    private static final int K = 5;
    private static final Path RESOURCES_DIR = Path.of("src", "main", "resources");
    private static final Path OBSERVED_DATA_FILE = RESOURCES_DIR.resolve(
            Path.of("inputs", "org", "sandwood", "benchmarking", "observedData", "LDAK5", "observed-data.json"));

    public static Map<TestType, TestData> getInputs() {
        ObservedData observedData = readObservedData();

        Map<TestType, TestData> m = new HashMap<>();
        {
            TestData t = new TestData();
            t.inputs.put("M", requireInteger(observedData.M, "M"));
            t.inputs.put("wObserved", toZeroIndexed(requireIntArray(observedData.w, "w")));
            t.inputs.put("doc", toZeroIndexed(requireIntArray(observedData.doc, "doc")));
            t.inputs.put("alpha", requireArray(observedData.alpha, "alpha"));
            t.inputs.put("beta", requireArray(observedData.beta, "beta"));
            t.args = new String[] { "M", "wObserved", "doc", "alpha", "beta" };
            t.outputNames = new String[] { "theta", "phi" };
            m.put(TestType.Gibbs, t);
        }
        return m;
    }

    private static ObservedData readObservedData() {
        try(InputStream in = openObservedData()) {
            ObjectMapper mapper = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
                    false);
            ObservedData observedData = mapper.readValue(in, ObservedData.class);
            requireInteger(observedData.V, "V");
            requireInteger(observedData.M, "M");
            requireInteger(observedData.N, "N");
            requireIntArray(observedData.w, "w");
            requireIntArray(observedData.doc, "doc");
            requireArray(observedData.alpha, "alpha");
            requireArray(observedData.beta, "beta");
            validateDimensions(observedData);
            validatePositive(observedData.alpha, "alpha");
            validatePositive(observedData.beta, "beta");
            validateRange(observedData.w, 1, observedData.V, "w");
            validateRange(observedData.doc, 1, observedData.M, "doc");
            return observedData;
        } catch(IOException e) {
            throw new IllegalStateException("Failed to read observed data from " + observedDataResource(), e);
        }
    }

    private static InputStream openObservedData() throws IOException {
        String resourceName = observedDataResource();
        InputStream in = LDAK5.class.getClassLoader().getResourceAsStream(resourceName);
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

    private static void validateDimensions(ObservedData observedData) {
        if(observedData.alpha.length != K) {
            throw new IllegalStateException("LDAK5 expects alpha length " + K + " in " + observedDataResource()
                    + ", but alpha has length " + observedData.alpha.length);
        }
        if(observedData.beta.length != observedData.V) {
            throw new IllegalStateException("Observed data dimension mismatch in " + observedDataResource()
                    + ": V is " + observedData.V + " but beta has length " + observedData.beta.length);
        }
        if(observedData.w.length != observedData.N || observedData.doc.length != observedData.N) {
            throw new IllegalStateException("Observed data length mismatch in " + observedDataResource()
                    + ": N is " + observedData.N + ", w has length " + observedData.w.length
                    + ", and doc has length " + observedData.doc.length);
        }
    }

    private static int[] toZeroIndexed(int[] values) {
        int[] zeroIndexed = new int[values.length];
        for(int i = 0; i < values.length; i++) {
            zeroIndexed[i] = values[i] - 1;
        }
        return zeroIndexed;
    }

    private static void validateRange(int[] values, int min, int max, String name) {
        for(int i = 0; i < values.length; i++) {
            if(values[i] < min || values[i] > max) {
                throw new IllegalStateException("Observed data array \"" + name + "\" has value " + values[i]
                        + " at index " + i + " outside expected range [" + min + ", " + max + "] in "
                        + observedDataResource());
            }
        }
    }

    private static void validatePositive(double[] values, String name) {
        for(int i = 0; i < values.length; i++) {
            if(values[i] <= 0.0) {
                throw new IllegalStateException("Observed data array \"" + name + "\" has non-positive value "
                        + values[i] + " at index " + i + " in " + observedDataResource());
            }
        }
    }

    private static int requireInteger(Integer value, String name) {
        if(value == null) {
            throw new IllegalStateException(
                    "Missing observed data integer \"" + name + "\" in " + observedDataResource());
        }
        return value;
    }

    private static int[] requireIntArray(int[] value, String name) {
        if(value == null) {
            throw new IllegalStateException(
                    "Missing observed data array \"" + name + "\" in " + observedDataResource());
        }
        return value;
    }

    private static double[] requireArray(double[] value, String name) {
        if(value == null) {
            throw new IllegalStateException(
                    "Missing observed data array \"" + name + "\" in " + observedDataResource());
        }
        return value;
    }

    private static class ObservedData {
        public Integer V;
        public Integer M;
        public Integer N;
        public int[] w;
        public int[] doc;
        public double[] alpha;
        public double[] beta;
    }
}