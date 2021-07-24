package com.github.jelmerk.knn.hnsw;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
//import java.util.concurrent.ExecutorService;
//import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.SearchResult;

/**
 * Example application, inserts them into an hnsw index and lets you query them.
 */
public class EmbText {

    public static void main(String[] args) throws Exception {

        String embPath = args[0];
        int topK = Integer.parseInt(args[1]);
        int nb = Integer.parseInt(args[2]);
        int ef = Integer.parseInt(args[3]);
        int embDim = Integer.parseInt(args[4]);
//        int samples = Integer.parseInt(args[4]);
        String outPutPath = args[5];
        String[] vec = args[6].split(",");
        float[] userEmb = new float[vec.length];
        for (int i = 0; i < vec.length; i ++){
            userEmb[i] = Float.parseFloat(vec[i]);
        }
//        int threadSize = Integer.parseInt(args[4]);
//        int perThreadNums = Integer.parseInt(args[5]);

        File embFile = new File(embPath);

        List<Word> goods = loadWordVectors(embFile);
//        Collections.shuffle(goods);
//        List<Word> rr = goods.subList(0, samples);
//        List<String> ids = rr.stream().map(Word::id).collect(Collectors.toList());

        System.out.println("Constructing index.");

        HnswIndex<String, float[], Word, Float> hnswIndex = HnswIndex
                .newBuilder(embDim, new CosinDistance(), goods.size())
                .withM(nb)
                .withEf(ef)
                .withEfConstruction(ef)
                .build();

        long start = System.currentTimeMillis();

        hnswIndex.addAll(goods, (workDone, max) -> System.out.printf("Added %d out of %d goods to the index.%n", workDone, max));

        long end = System.currentTimeMillis();

        long duration = end - start;

        System.out.printf("Creating index with %d goods took %d millis which is %d minutes.%n", hnswIndex.size(), duration, TimeUnit.MILLISECONDS.toMinutes(duration));

//        Index<String, float[], Word, Float> groundTruthIndex = hnswIndex.asExactIndex();

//        float[] userEmb = new float[] {0.0798802823f, -0.105617099f, -0.0222556125f, -0.238502607f, -0.105800487f, 0.0462308191f, 0.227768853f, -0.24982892f,
//                -0.148135915f, 0.0207138341f, -0.0645692348f, -0.0227155518f, -0.154298872f, 0.0410800651f, 0.206697628f, 0.213482797f,
//                0.147008881f, 0.137353539f, 0.120455682f, 0.0193726029f, 0.0719488785f, 0.0200199094f, 0.0307276808f, -0.1585152f, -0.00606888f,
//                0.0405771881f, 0.0369572341f, 0.0454871468f, -0.130782798f, 0.146907359f, -0.131885961f, 0.00772275496f, 0.202325612f,
//                0.0562758744f, -0.0459487215f, 0.10894084f, 0.175470695f, -0.0418495126f, 0.435695231f, -0.0582940616f, -0.0998019129f,
//                0.0567423292f, 0.00489831716f, -0.0351954438f, 0.0041315183f, 0.0592247061f, 0.0857765824f, -0.264579058f, -0.0413251184f,
//                -0.041364748f, 0.0934605f, 0.150109679f, 0.143055469f, 0.0328333527f, 0.0739659816f, 0.0115726292f, -0.0409326665f, 0.148354232f,
//                0.107371457f, 0.00305166561f, -0.0950880647f, 0.0697333887f, 0.131638855f, 0.0294979289f};

        userEmb = VectorUtils.normalize(userEmb);
//        创建线程池
//        ExecutorService executorService = Executors.newFixedThreadPool(threadSize);
//        long start_time = System.currentTimeMillis();
//
//        for (int j = 0; j < threadSize; j++) {
//
//            executorService.execute(new TestPerformance(hnswIndex, userEmb, topK, perThreadNums));
//        }
////        等线程全部执行完后关闭线程池
//        executorService.shutdown();
//        executorService.awaitTermination(Integer.MAX_VALUE, TimeUnit.DAYS);
//        long end_time = System.currentTimeMillis();
//        System.out.println("测试次数：" + TestPerformance.atomicInteger.get());
//        System.out.println("用时：" + (end_time - start_time));
//        System.out.println("速度：" + TestPerformance.atomicInteger.get() * 1000 / (end_time - start_time) + "次/秒");

        List<SearchResult<Word, Float>> approximateResults = hnswIndex.findNearest(userEmb, topK);

        StringBuilder sb = new StringBuilder(); // 单线程更高效，线程不安全
        for (SearchResult<Word, Float> sr : approximateResults)
            sb.append(sr.item().id()).append(",");
        sb.deleteCharAt(sb.length() - 1);

        BufferedWriter bw = new BufferedWriter(new FileWriter(outPutPath, false));
        bw.write(sb.toString());
        bw.flush();

//        System.out.printf("saved top%d goods_id to %s\n", topK, outPutPath);

//        double res = 0;
//        int n = 1;
//        for (String id: ids){
//
//            List<SearchResult<Word, Float>> approximateResults = hnswIndex.findNeighbors(id, topK);
//
//            long t1 = System.currentTimeMillis();
//            List<SearchResult<Word, Float>> groundTruthResults = groundTruthIndex.findNeighbors(id, topK);
//            long t2 = System.currentTimeMillis();

//            System.out.println("Most similar goods found using HNSW index : %n%n");
//
//            for (SearchResult<Word, Float> result : approximateResults) {
//                System.out.printf("%s %.4f%n", result.item().id(), result.distance());
//            }
//
//            System.out.printf("%nMost similar goods found using exact index: %n%n");
//
//            for (SearchResult<Word, Float> result : groundTruthResults) {
//                System.out.printf("%s %.4f%n", result.item().id(), result.distance());

//            int correct = groundTruthResults.stream().mapToInt(r -> approximateResults.contains(r) ? 1 : 0).sum();
//            double rrt = correct / (double) groundTruthResults.size();
//            System.out.printf("%none time Accuracy : %.4f, i = %d%n", rrt, n);
//            System.out.println("bruteforce one time: " + (t2 - t1) / 1000.0);
//            res += rrt;
//            n += 1;
        }
//        System.out.println("avg Accuracy: " + res / (double) ids.size());
//    }


    private static List<Word> loadWordVectors(File path) throws IOException {
        System.out.printf("Loading goods from %s%n", path);

        BufferedReader br = new BufferedReader(new FileReader(path));
        String contentLine ;
        List<Word> list = new ArrayList<>();
        while ((contentLine = br.readLine()) != null){
            String[] tokens = contentLine.split("\\t");

            String item = tokens[0];
            String[] embed = tokens[1].split(",");

            float[] vector = new float[embed.length];
            for (int i = 0; i < vector.length - 1; i++) {
                vector[i] = Float.parseFloat(embed[i]);
            }
            list.add(new Word(item, VectorUtils.normalize(vector)));
        }
        return list;
    }
}




