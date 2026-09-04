/*
 * Sandwood
 *
 * Copyright (c) 2019-2023, Oracle and/or its affiliates
 *
 * Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
 */

package org.sandwood.benchmarking.tests.util;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.PriorityQueue;

import org.sandwood.benchmarking.tests.exceptions.SandwoodTestException;
import org.sandwood.runtime.internal.json.JsonEncoder;

public class JsonResultsEncoder {

    public static final String metadataName = "metadata";

    public static void writeData(String filename, Map<String, Object> outputs) throws IOException {
        File file = new File(filename);
        File dir = file.getParentFile();
        dir.mkdirs();
        JsonEncoder e = new JsonEncoder(file);

        e.addStartObject();
        PriorityQueue<String> p = new PriorityQueue<>(outputs.keySet());
        while(!p.isEmpty()) {
            String name = p.poll();
            Object o = outputs.get(name);
            e.addField(name);
            if(name.equals(metadataName))
                writeMetadata(e, o);
            else {
                e.addStartObject();
                e.addString("type", getType(o));
                e.addObject("value", o);
                e.addEndObject();
            }
        }

        e.addEndObject();
        e.close();
    }

    private static void writeMetadata(JsonEncoder e, Object metadata) throws IOException {
        if(!(metadata instanceof Map<?, ?>))
            throw new SandwoodTestException("Metadata must be a map");

        writePlainObject(e, (Map<?, ?>) metadata);
    }

    private static void writePlainObject(JsonEncoder e, Map<?, ?> map) throws IOException {
        e.addStartObject();
        PriorityQueue<String> p = getSortedKeys(map);

        while(!p.isEmpty()) {
            String name = p.poll();
            Object value = map.get(name);
            if(value instanceof Integer)
                e.addInt(name, (Integer) value);
            else if(value instanceof Double)
                e.addDouble(name, (Double) value);
            else if(value instanceof Boolean)
                e.addBoolean(name, (Boolean) value);
            else if(value instanceof String)
                e.addString(name, (String) value);
            else if(value instanceof Map<?, ?>) {
                e.addField(name);
                writePlainObject(e, (Map<?, ?>) value);
            } else
                throw new SandwoodTestException("Unknown metadata type");
        }
        e.addEndObject();
    }

    private static PriorityQueue<String> getSortedKeys(Map<?, ?> metadataMap) {
        PriorityQueue<String> p = new PriorityQueue<>();
        for(Object key: metadataMap.keySet()) {
            if(!(key instanceof String))
                throw new SandwoodTestException("Metadata keys must be strings");
            p.add((String) key);
        }
        return p;
    }

    private static String getType(Object o) {
        if(o instanceof Integer)
            return "int";
        else if(o instanceof Double)
            return "double";
        else if(o instanceof Boolean)
            return "boolean";
        else if(o instanceof Object[])
            return getType(((Object[]) o)[0]) + "[]"; // This will fail for empty arrays.
        else if(o instanceof double[])
            return "double[]";
        else if(o instanceof int[])
            return "int[]";
        else if(o instanceof boolean[])
            return "boolean[]";

        throw new SandwoodTestException("Unknown type");
    }
}
