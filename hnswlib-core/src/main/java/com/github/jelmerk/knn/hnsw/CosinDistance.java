package com.github.jelmerk.knn.hnsw;

public class CosinDistance implements DistanceFunction<float[], Float> {

    private static final long serialVersionUID = 1L;

    @Override
    public Float distance(float[] u, float[] v) {
        float dot = 0.0f;
        float nru = 0.0f;
        float nrv = 0.0f;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
            nru += u[i] * u[i];
            nrv += v[i] * v[i];
        }

        float similarity = dot / (float)(Math.sqrt(nru) * Math.sqrt(nrv));
        return 1 - similarity;
    }
}
