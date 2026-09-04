/*
 * Sandwood
 *
 * Copyright (c) 2019-2025, Oracle and/or its affiliates
 *
 * Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
 */

package org.sandwood.benchmarking.driver;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import org.sandwood.compiler.CompilationOptions;
import org.sandwood.compiler.compilation.ChildFirstClassLoader;
import org.sandwood.compiler.compilation.util.CompileUtils;
import org.sandwood.compiler.compilation.util.SandwoodCompileError;
import org.sandwood.compiler.dataflowGraph.Sandwood;
import org.sandwood.runtime.exceptions.SandwoodJsonException;
import org.sandwood.runtime.model.ExecutionTarget;
import org.sandwood.runtime.model.Model;
import org.sandwood.runtime.model.RetentionPolicy;
import org.sandwood.runtime.model.variables.ObservedBoolean;
import org.sandwood.runtime.model.variables.ObservedBooleanArray;
import org.sandwood.runtime.model.variables.ObservedDouble;
import org.sandwood.runtime.model.variables.ObservedDoubleArray;
import org.sandwood.runtime.model.variables.ObservedInteger;
import org.sandwood.runtime.model.variables.ObservedIntegerArray;
import org.sandwood.runtime.model.variables.ObservedObjectArray;

import org.sandwood.benchmarking.tests.exceptions.SandwoodTestException;
import org.sandwood.benchmarking.tests.util.CompilerState;
import org.sandwood.benchmarking.tests.util.JsonResultsDecoder;
import org.sandwood.benchmarking.tests.util.JsonResultsEncoder;
import org.sandwood.benchmarking.tests.util.TestStringGenerator;

public class SandwoodBenchmarkDriver {
    private interface ResultGetter {
        Object get(String resultName)
                throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException;
    }

    public enum TestType {
        Gibbs,
        EvidenceLog,
        Forward
    }

    public static class TestData {
        public final Map<String, Object> inputs = new HashMap<>();
        public final Set<String> shapes = new HashSet<>();
        // Args contains the key values of inputs allowing the use of constructors.
        public String[] args;
        public String[] outputNames;
    }

    public static class TestDesc {
        public final ExecutionTarget executionTarget;
        public final String filename;
        public final String modelName;
        public final String packageName;
        public final Map<TestType, TestData> inputData;
        public final long seed;
        public final int numChains;

        public TestDesc(ExecutionTarget target, String filename, Map<TestType, TestData> testData, long seed,
                int numChains) {
            this.executionTarget = target;
            this.filename = filename;
            this.inputData = testData;
            this.seed = seed;
            this.numChains = numChains;

            String[] parts = filename.split("\\.")[0].split(Pattern.quote(File.separator));
            int length = parts.length;

            modelName = parts[length - 1];

            StringBuilder sb = new StringBuilder();
            if(parts.length > 3)
                sb.append(parts[4]);
            for(int i = 5; i < length - 1; i++) {
                sb.append(".");
                sb.append(parts[i]);
            }
            packageName = sb.toString();
        }

        @Override
        public String toString() {
            return modelName + " " + (executionTarget == null ? "Default" :executionTarget);
        }
    }

    private static final String baseDir = "src" + File.separator + "main" + File.separator + "resources";
    private static final String sourceDir = baseDir + File.separator + "inputs";
    private static final String outputDir = baseDir + File.separator + "expectedOutputs";

    private static final int inferenceSteps = 1000;
    private static final int burnInSteps = 1500;
    private static final int probSteps = 1000;
    private static final int forwardSteps = 1000;

    private static final String runningTimeSecondsName = "runningTimeSeconds";

    private ChildFirstClassLoader cl;

    protected static final boolean constructingJson = true;
    private final TestStringGenerator tsg = new TestStringGenerator();

    public static void main(String[] args) throws Exception {
        SandwoodBenchmarkDriver d = new SandwoodBenchmarkDriver();
        int numChains = 10;
        d.runBenchmarks(numChains);
    }

    private void runBenchmarks(int numChains) throws Exception {
        for(TestDesc test: getTargets(numChains)) {
            CompilerState.reset();
            File targetDir = Files.createTempDirectory(null).toFile();
            compileModel(test, targetDir);

            runModelTests(test, targetDir.getAbsolutePath());

            cleanup(targetDir);
        }
    }

    protected Map<String, Object> getOutputs(TestType testType, TestDesc test)
            throws SandwoodJsonException, IOException {
        return getOutputs(testType, test, null);
    }

    protected Map<String, Object> getOutputs(TestType testType, TestDesc test, String tag)
            throws SandwoodJsonException, IOException {
        String filename = getFilename(testType, test.executionTarget, test.modelName, tag);
        return JsonResultsDecoder.readData(filename);
    }

    private void writeOutputs(TestType testType, TestDesc test, Map<String, Object> outputs) throws IOException {
        writeOutputs(testType, test, null, outputs);
    }

    private void writeOutputs(TestType testType, TestDesc test, String tag, Map<String, Object> outputs)
            throws IOException {
        if(test.executionTarget != null) {
            String filename = getFilename(testType, test.executionTarget, test.modelName, tag);
            JsonResultsEncoder.writeData(filename, outputs);
        }
    }

    private String getFilename(TestType testType, ExecutionTarget target, String modelName, String tag) {
        String[] parts = modelName.split(Pattern.quote(File.separator));
        modelName = parts[parts.length - 1].split("\\.")[0];
        String filename = tsg.getFilename(this, outputDir, modelName,
                (target == null ? ExecutionTarget.singleThread :target).executionType,
                testType.toString() + (tag == null ? "" :tag), "json");
        return filename;
    }

    private List<TestDesc> getTargets(int numChains) {
        List<ExecutionTarget> executionTargets = getExecutionTargets();
        Map<String, Map<TestType, TestData>> modelTargets = getModelTargets();
        List<TestDesc> targets = new ArrayList<>();
        for(String filename: modelTargets.keySet()) {
            for(ExecutionTarget e: executionTargets) {
                // Restrict the set of execution targets considered in benchmarking. For now, we consider SingleThread.
                if(e == null || e.toString().equals("MultiThread") || e.toString().equals("ForkJoin")) {
                    continue;
                }

                for(int chainIndex = 0; chainIndex < numChains; chainIndex++) {
                    // The PRNG seeds will be 42, 43, ..., 42 + numChains -1.
                    long seed = 42 + chainIndex;
                    targets.add(new TestDesc(e, filename, modelTargets.get(filename), seed, numChains));
                }
            }
        }
        return targets;
    }

    protected Map<String, Map<TestType, TestData>> getModelTargets() {
        Map<String, Map<TestType, TestData>> modelData = new LinkedHashMap<>();
        String packageDir = TestStringGenerator.getPackageDir(this, sourceDir);

        File dir = new File(packageDir);
        assert dir.isDirectory();
        String[] files = dir.list((dir1, name) -> name.endsWith(".sandwood"));

        // Remove some benchmark programs. This is useful during the development of this code repository because we
        // would like not to run all benchmarks every time we would like to test the execution.
        String[] filesToFilterOut = { "LDAK5.sandwood", "HMMExample.sandwood", "LowDimMix.sandwood",
                "EightSchoolsCentered.sandwood", "EightSchoolsNonCentered.sandwood", "Earnings.sandwood",
                "NES.sandwood", "SBLRC.sandwood", "ARK.sandwood", "GARCH.sandwood",
                "Flip1CoinMK0.sandwood", "SequentialTrajectories.sandwood" };
        Set<String> filesToFilterOutSet = new HashSet<>(Arrays.asList(filesToFilterOut));
        files = Arrays.stream(files).filter(file -> !filesToFilterOutSet.contains(file)).toArray(String[]::new);

        Arrays.sort(files);

        for(String filename: files) {
            try {
                String packageName = this.getClass().getPackageName();
                String classname = filename.split("\\.")[0];
                Class<?> c = Class.forName(packageName + ".compileAndRunInputs." + classname);
                @SuppressWarnings("unchecked")
                Map<TestType, TestData> t = (Map<TestType, TestData>) c.getMethod("getInputs").invoke(null);
                modelData.put(packageDir + File.separator + filename, t);
            } catch(Exception e) {
            }
        }
        return modelData;
    }

    private static List<ExecutionTarget> getExecutionTargets() {
        List<ExecutionTarget> executionTargets = new ArrayList<>();
        executionTargets.add(null);
        for(ExecutionTarget e: ExecutionTarget.implementedTargets)
            executionTargets.add(e);
        return executionTargets;
    }

    private void compileModel(TestDesc testDesc, File targetDir) throws IOException {
        CompilationOptions opts = new CompilationOptions();
        opts.setTargetDirectory(targetDir.getAbsolutePath());
        opts.fullInferenceRequired(false);
        opts.setCalculateIndividualProbabilities();
        // System.out.println("Compiling to: " + opts.getTargetDirectory());
        opts.setModelPath(sourceDir);
        opts.setModelFile(testDesc.filename);
        List<SandwoodCompileError> errors = Sandwood.compileModel(opts).getErrors();
        assert CompileUtils.parseErrors(errors).equals("");
    }

    private void runModelTests(TestDesc testDesc, String compiledDir) throws Exception {
        System.out.println("Execution target: " + testDesc.executionTarget);

        for(TestType test: testDesc.inputData.keySet()) {
            switch(test) {
                case Gibbs: {
                    Map<String, Object> outputs = constructingJson ? null :getOutputs(test, testDesc, "Sample");
                    testGibbsConstructor(testDesc, compiledDir, outputs);
                    outputs = constructingJson ? null :getOutputs(test, testDesc, "Map");
                    testGibbsVariableSet(testDesc, compiledDir, outputs);
                    break;
                }
                case EvidenceLog: {
                    Map<String, Object> outputs = constructingJson ? null :getOutputs(test, testDesc);
                    testLogEvidenceConstructor(testDesc, compiledDir, outputs);
                    break;
                }
                case Forward: {
                    Map<String, Object> outputs = constructingJson ? null :getOutputs(test, testDesc);
                    testForwardConstructor(testDesc, compiledDir, outputs);
                    break;
                }
            }
        }
    }

    private void testForwardConstructor(TestDesc test, String compiledDir, Map<String, Object> outputs)
            throws Exception {
        TestData data = test.inputData.get(TestType.Forward);
        Class<?> c = getClass(compiledDir, test);
        Model model = getModel(c, test, data);

        assert model.readyExecute();

        // Call the backward pass method
        long start = System.nanoTime();
        model.execute(forwardSteps);
        long end = System.nanoTime();
        double runningTimeSeconds = getRunningTimeSeconds(start, end);
        System.out.print(test.modelName + " Forward Execution of " + forwardSteps + " " + runningTimeSeconds);

        if(constructingJson) {
            System.out.println();
            outputs = new HashMap<>();
            outputs.put(JsonResultsEncoder.metadataName, getMetadata(test, runningTimeSeconds));
            for(String resultName: data.outputNames) {
                Object generated = getSamplesValue(c, model, resultName);
                outputs.put(resultName, generated);
            }
            writeOutputs(TestType.Forward, test, outputs);
        } else if(compare(outputs, (String name) -> getSamplesValue(c, model, name)))
            System.out.println("\t\tPass");
        else
            System.out.println("\t\t**** Fail ****");

        model.close();
    }

    private void testLogEvidenceConstructor(TestDesc test, String compiledDir, Map<String, Object> outputs)
            throws Exception {
        TestData data = test.inputData.get(TestType.EvidenceLog);
        Class<?> c = getClass(compiledDir, test);
        Model model = getModel(c, test, data);

        assert model.readyInferProbabilities();

        // Call the backward pass method
        long start = System.nanoTime();
        model.inferProbabilities(probSteps);
        long end = System.nanoTime();
        double runningTimeSeconds = getRunningTimeSeconds(start, end);
        System.out.print(test.modelName + " Probability Inference of " + probSteps + " " + runningTimeSeconds);

        if(constructingJson) {
            System.out.println();
            outputs = new HashMap<>();
            outputs.put(JsonResultsEncoder.metadataName, getMetadata(test, runningTimeSeconds));
            outputs.put("modelProbability", model.getLogProbability());
            for(String resultName: data.outputNames) {
                Object generated = getLogProbability(c, model, resultName);
                outputs.put(resultName, generated);
            }
            writeOutputs(TestType.EvidenceLog, test, outputs);
        } else {
            assert outputs.get("modelProbability").equals(model.getLogProbability());
            if(compare(outputs, data.outputNames, (String name) -> getLogProbability(c, model, name)))
                System.out.println("\t\tPass");
            else
                System.out.println("\t\t**** Fail ****");
        }

        model.close();
    }

    private Object getLogProbability(Class<?> c, Model model, String resultName)
            throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        Field f = c.getField(resultName);
        Object outputField = f.get(model);
        Method outputMethod = outputField.getClass().getMethod("probabilityComputed");
        assert (boolean) outputMethod.invoke(outputField, new Object[0]);
        outputMethod = outputField.getClass().getMethod("getLogProbability");
        return outputMethod.invoke(outputField, new Object[0]);
    }

    private void testGibbsVariableSet(TestDesc test, String compiledDir, Map<String, Object> outputs) throws Exception {
        Class<?> c = getClass(compiledDir, test);
        Model model = getModel(c, test);

        model.setDefaultRetentionPolicy(RetentionPolicy.MAP);

        TestData data = test.inputData.get(TestType.Gibbs);

        assert !model.readyInferValues() || data.args.length == 0;

        for(String inputName: data.inputs.keySet()) {
            setValue(model, c, inputName, data.inputs.get(inputName));
        }

        assert model.readyInferValues();

        // Call the backward pass method
        long start = System.nanoTime();
        model.inferValues(inferenceSteps, burnInSteps, 0);
        long end = System.nanoTime();
        double runningTimeSeconds = getRunningTimeSeconds(start, end);
        System.out.print(
                test.modelName + " Infer map of " + inferenceSteps + " seed " + test.seed + " " + runningTimeSeconds);

        if(constructingJson) {
            System.out.println();
            outputs = new HashMap<>();
            outputs.put(JsonResultsEncoder.metadataName, getMetadata(test, runningTimeSeconds));
            for(String resultName: data.outputNames) {
                Object generated = getMapValue(c, model, resultName);
                outputs.put(resultName, generated);
            }
            writeOutputs(TestType.Gibbs, test, "Map" + test.seed, outputs);
        } else {
            // Call the get results method
            if(compare(outputs, (String name) -> getMapValue(c, model, name)))
                System.out.println("\t\tPass");
            else
                System.out.println("\t\t**** Fail ****");
        }

        model.close();
    }

    private void testGibbsConstructor(TestDesc test, String compiledDir, Map<String, Object> outputs) throws Exception {
        Class<?> c = getClass(compiledDir, test);

        TestData data = test.inputData.get(TestType.Gibbs);
        Model model = getModel(c, test, data);

        assert model.readyInferValues();

        // Call the backward pass method
        long start = System.nanoTime();
        model.inferValues(inferenceSteps, burnInSteps, 0);
        long end = System.nanoTime();
        double runningTimeSeconds = getRunningTimeSeconds(start, end);
        System.out.print(test.modelName + " Infer sample of " + inferenceSteps + " seed " + test.seed + " "
                + runningTimeSeconds);

        if(constructingJson) {
            System.out.println();
            outputs = new HashMap<>();
            outputs.put(JsonResultsEncoder.metadataName, getMetadata(test, runningTimeSeconds));
            for(String resultName: data.outputNames) {
                Object generated = getSamplesValue(c, model, resultName);
                outputs.put(resultName, generated);
            }
            writeOutputs(TestType.Gibbs, test, "Sample" + test.seed, outputs);
        } else {
            // Call the get results method
            if(compare(outputs, (String name) -> getSamplesValue(c, model, name)))
                System.out.println("\t\tPass");
            else
                System.out.println("\t\t**** Fail ****");
        }

        model.close();
    }

    private boolean compare(Map<String, Object> outputs, ResultGetter result) throws Exception {
        List<String> outputNames = new ArrayList<>();
        for(String outputName: outputs.keySet()) {
            if(!isMetadataOutput(outputName))
                outputNames.add(outputName);
        }
        return compare(outputs, outputNames, result);
    }

    private boolean compare(Map<String, Object> outputs, String[] valueNames, ResultGetter result) throws Exception {
        List<String> listNames = new ArrayList<>();
        for(String s: valueNames)
            listNames.add(s);
        return compare(outputs, listNames, result);
    }

    private boolean compare(Map<String, Object> outputs, Collection<String> valueNames, ResultGetter result)
            throws Exception {
        for(String resultName: valueNames) {
            if(isMetadataOutput(resultName))
                continue;
            Object expected = outputs.get(resultName);
            Object generated = result.get(resultName);
            if(!compare(expected, generated))
                return false;
        }
        return true;
    }

    private boolean isMetadataOutput(String outputName) {
        return outputName.equals(JsonResultsEncoder.metadataName) || outputName.equals(runningTimeSecondsName);
    }

    private double getRunningTimeSeconds(long start, long end) {
        return (end - start) / 1_000_000_000.0;
    }

    private Map<String, Object> getMetadata(TestDesc test, double runningTimeSeconds) {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put(runningTimeSecondsName, runningTimeSeconds);
        metadata.put("samplingConfiguration", getSamplingConfiguration(test));
        return metadata;
    }

    private Map<String, Object> getSamplingConfiguration(TestDesc test) {
        Map<String, Object> samplingConfiguration = new HashMap<>();
        samplingConfiguration.put("numChains", test.numChains);
        samplingConfiguration.put("numWarmupsPerChain", burnInSteps);
        samplingConfiguration.put("numPosteriorSamplesPerChain", inferenceSteps);
        return samplingConfiguration;
    }

    private Object getSamplesValue(Class<?> c, Model model, String resultName)
            throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        Field f = c.getField(resultName);
        Object outputField = f.get(model);

        Class<?> fieldClass = outputField.getClass();
        Method outputMethod = fieldClass.getMethod("getSamples");
        return outputMethod.invoke(outputField, new Object[0]);
    }

    private Object getMapValue(Class<?> c, Model model, String resultName)
            throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        Field f = c.getField(resultName);
        Object outputField = f.get(model);

        Class<?> fieldClass = outputField.getClass();
        Method outputMethod = fieldClass.getMethod("getMAP");

        return outputMethod.invoke(outputField, new Object[0]);
    }

    private void setValue(Model model, Class<?> c, String inputName, Object input) throws Exception {
        Field f = c.getField(inputName);
        Object inputField = f.get(model);
        if(inputField instanceof ObservedDouble) {
            ((ObservedDouble) inputField).setValue((Double) input);
        } else if(inputField instanceof ObservedDoubleArray) {
            ((ObservedDoubleArray) inputField).setValue((double[]) input);
        } else if(inputField instanceof ObservedInteger) {
            ((ObservedInteger) inputField).setValue((Integer) input);
        } else if(inputField instanceof ObservedIntegerArray) {
            ((ObservedIntegerArray) inputField).setValue((int[]) input);
        } else if(inputField instanceof ObservedBoolean) {
            ((ObservedBoolean) inputField).setValue((Boolean) input);
        } else if(inputField instanceof ObservedBooleanArray) {
            ((ObservedBooleanArray) inputField).setValue((boolean[]) input);
        } else if(inputField instanceof ObservedObjectArray<?>) {
            for(Method m: inputField.getClass().getMethods()) {
                if(m.getName().equals("setValue")) {
                    m.invoke(inputField, input);
                    break;
                }
            }
        } else
            throw new SandwoodTestException("Unknown input type");
    }

    private Class<?> getClass(String dir, TestDesc desc) throws ClassNotFoundException, MalformedURLException {
        // Skip the first 3 elements as they are not part of the package.
        String className = desc.packageName + "." + desc.modelName;

        // Construct a URL[] containing the output directory.
        File file = new File(dir);
        URL url = file.toURI().toURL();
        URL[] urls = new URL[] { url };

        // Load the contents
        cl = new ChildFirstClassLoader(urls);
        // Load the class
        return cl.loadClass(className);
    }

    private Model getModel(Class<?> cls, TestDesc test)
            throws NoSuchMethodException, InstantiationException, IllegalAccessException, IllegalArgumentException,
            InvocationTargetException {
        return getModel(cls, test, null);
    }

    private Model getModel(Class<?> cls, TestDesc test, TestData data)
            throws NoSuchMethodException, InstantiationException, IllegalAccessException, IllegalArgumentException,
            InvocationTargetException {
        ExecutionTarget target = test.executionTarget;
        Class<?>[] types;
        Object[] args;

        if(data != null) {
            int noArgs = data.args.length;
            types = new Class<?>[noArgs];
            args = new Object[noArgs];

            for(int i = 0; i < noArgs; i++) {
                Object arg = data.inputs.get(data.args[i]);
                args[i] = arg;
                types[i] = arg.getClass();
            }
            insertBaseTypes(types);

        } else {
            types = new Class<?>[0];
            args = new Object[0];
        }

        Constructor<?> cons = cls.getConstructor(types);
        Model m = (Model) cons.newInstance(args);
        if(target != null) {
            m.setExecutionTarget(target);
            switch(target.executionType) {
                case SingleThreadCPU:
                    m.setThreadCount(1);
                    break;
                case MultiThreadCPU:
                    //                    m.setThreadCount(6);
                    break;
                default:
                    throw new SandwoodTestException("Unknown execution target " + target);
            }
        }

        // Call initializeSeed(42)
        m.initializeSeed(test.seed);

        return m;
    }

    private void insertBaseTypes(Class<?>[] types) {
        for(int i = 0; i < types.length; i++) {
            Class<?> c = types[i];
            if(c.equals(Integer.class))
                types[i] = int.class;
            else if(c.equals(Double.class))
                types[i] = double.class;
            else if(c.equals(Boolean.class))
                types[i] = boolean.class;
        }
    }

    private void cleanup(File targetDir) throws IOException {
        // close the class loader if it was opened.
        if(cl != null) {
            cl.close();
            cl = null;
        }
        if(targetDir != null) {
            deleteFile(targetDir);
            targetDir = null;
        }
    }

    private void deleteFile(File file) {
        if(file.exists()) {
            if(file.isDirectory()) {
                for(File subFile: file.listFiles())
                    deleteFile(subFile);
            }
            file.delete();
        }
    }

    // Method to test results;
    private boolean compare(Object expected, Object generated) throws Exception {
        if(expected instanceof Double) {
            double e = ((Double) expected).doubleValue();
            double g = ((Double) generated).doubleValue();
            return e == g;
        } else if(expected instanceof double[]) {
            double[] e = (double[]) expected;
            double[] g = (double[]) generated;
            // we only check the entries stored as might be returning a lot of entries.
            for(int i = 0; i < e.length && i < g.length; i++)
                if(e[i] != g[i])
                    return false;
            return true;
        } else if(expected instanceof Integer) {
            int e = ((Integer) expected).intValue();
            int g = ((Integer) generated).intValue();
            return e == g;
        } else if(expected instanceof int[]) {
            int[] e = (int[]) expected;
            int[] g = (int[]) generated;
            // we only check the entries stored as might be returning a lot of entries.
            for(int i = 0; i < e.length && i < g.length; i++)
                if(e[i] != g[i])
                    return false;
            return true;
        } else if(expected instanceof Boolean) {
            boolean e = ((Boolean) expected).booleanValue();
            boolean g = ((Boolean) generated).booleanValue();
            return e == g;
        } else if(expected instanceof boolean[]) {
            boolean[] e = (boolean[]) expected;
            boolean[] g = (boolean[]) generated;
            // we only check the entries stored as might be returning a lot of entries.
            for(int i = 0; i < e.length && i < g.length; i++)
                if(e[i] != g[i])
                    return false;
            return true;
        } else if(expected instanceof Object[]) {
            Object[] e = (Object[]) expected;
            Object[] g = (Object[]) generated;
            if(e.length != g.length)
                return false;
            for(int i = 0; i < e.length; i++) {
                if(!compare(e[i], g[i]))
                    return false;
            }
            return true;
        } else
            throw new Exception("unknown return type");
    }
}
