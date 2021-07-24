package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.SearchResult;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class TestPerformance implements Runnable {

    //每个线程的执行次数
    private int size;
    private int topK;
    private HnswIndex<String, float[], Word, Float> index;
    private float[] emb;

    //记录多线程的总执行次数，保证高并发下的原子性
    public static AtomicInteger atomicInteger = new AtomicInteger(0);

    public TestPerformance(HnswIndex<String, float[], Word, Float> index, float[] arr, int topK, int size) {
        this.size = size;
        this.topK = topK;
        this.index = index;
        this.emb = arr;
    }

    @Override
    public void run() {

        int count = 0;
        while (count < this.size) {
            count++;

            atomicInteger.getAndIncrement();

            List<SearchResult<Word, Float>> approximateResults = this.index.findNearest(this.emb, this.topK);

            System.out.println("线程ID与对应的执行次数：" + Thread.currentThread().getId() + "--->" + count);
        }
    }
}
