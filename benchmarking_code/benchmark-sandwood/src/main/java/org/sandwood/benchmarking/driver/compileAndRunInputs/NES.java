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

public class NES {
    private static final Path RESOURCES_DIR = Path.of("src", "main", "resources");
    private static final Path OBSERVED_DATA_FILE = RESOURCES_DIR.resolve(
            Path.of("inputs", "org", "sandwood", "benchmarking", "observedData", "NES", "observed-data.json"));

    public static Map<TestType, TestData> getInputs() {
        ObservedData observedData = readObservedData();

        Map<TestType, TestData> m = new HashMap<>();
        {
            TestData t = new TestData();
            t.inputs.put("partyid7Observed", requireArray(observedData.partyid7, "partyid7"));
            t.inputs.put("real_ideo", requireArray(observedData.real_ideo, "real_ideo"));
            t.inputs.put("race_adj", requireArray(observedData.race_adj, "race_adj"));
            t.inputs.put("educ1", requireArray(observedData.educ1, "educ1"));
            t.inputs.put("gender", requireArray(observedData.gender, "gender"));
            t.inputs.put("income", requireArray(observedData.income, "income"));
            t.inputs.put("age_discrete", requireIntArray(observedData.age_discrete, "age_discrete"));
            t.args = new String[] { "partyid7Observed", "real_ideo", "race_adj", "educ1", "gender", "income",
                    "age_discrete" };
            t.outputNames = new String[] { "beta", "sigma" };
            m.put(TestType.Gibbs, t);
        }
        return m;
    }

    private static ObservedData readObservedData() {
        try(InputStream in = openObservedData()) {
            ObjectMapper mapper = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
                    false);
            ObservedData observedData = mapper.readValue(in, ObservedData.class);
            requireInteger(observedData.N, "N");
            requireIntArray(observedData.age_discrete, "age_discrete");
            requireArray(observedData.educ1, "educ1");
            requireArray(observedData.gender, "gender");
            requireArray(observedData.income, "income");
            requireArray(observedData.partyid7, "partyid7");
            requireArray(observedData.race_adj, "race_adj");
            requireArray(observedData.real_ideo, "real_ideo");
            validateLength(observedData);
            return observedData;
        } catch(IOException e) {
            throw new IllegalStateException("Failed to read observed data from " + observedDataResource(), e);
        }
    }

    private static InputStream openObservedData() throws IOException {
        String resourceName = observedDataResource();
        InputStream in = NES.class.getClassLoader().getResourceAsStream(resourceName);
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
        int n = observedData.N;
        if(observedData.age_discrete.length != n || observedData.educ1.length != n || observedData.gender.length != n
                || observedData.income.length != n || observedData.partyid7.length != n
                || observedData.race_adj.length != n || observedData.real_ideo.length != n)
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

    private static int[] requireIntArray(int[] value, String name) {
        if(value == null)
            throw new IllegalStateException("Missing observed data array \"" + name + "\" in "
                    + observedDataResource());
        return value;
    }

    private static class ObservedData {
        public Integer N;
        public double[] partyid7;
        public double[] real_ideo;
        public double[] race_adj;
        public double[] educ1;
        public double[] gender;
        public double[] income;
        public int[] age_discrete;
    }
}
