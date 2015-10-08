package finalproj;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class MahoutTryIt {
    
    public static final double[][] points = { {1, 1}, {2, 1}, {1, 2},
        {2, 2}, {3, 3}, {8, 8},
        {9, 8}, {8, 9}, {9, 9}};
    
    // Write data to sequence files in Hadoop (write the vector to sequence file)
    public static void writePointsToFile(List<Vector> points, String fileName,
            							FileSystem fs,Configuration conf) throws IOException {
        
                    Path path = new Path(fileName);
                    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, 
                    									LongWritable.class, VectorWritable.class);
                    long recNum = 0;
                    VectorWritable vec = new VectorWritable();
                    
                    for (Vector point : points) {
                        vec.set(point);
                        writer.append(new LongWritable(recNum++), vec);
                    }
                    
                    writer.close();
    }
    
    // Read the points to vector from 2D array
    public static List getPoints(double[][] raw) {
          List points = new ArrayList();
          for (int i = 0; i < raw.length; i++) {
            double[] fr = raw[i];
            Vector vec = new RandomAccessSparseVector(fr.length);
            vec.assign(fr);
            points.add(vec);
        }
          return points;
        }
    
    public static void main(String args[]) throws Exception {
        
        // specify the number of clusters 
        int k = 2;
        
        // read the values (features) - generate vectors from input data
          List vectors = getPoints(points);
          
          // Create input directories for data
          File testData = new File("testdata");
          
          if (!testData.exists()) {
            testData.mkdir();
          }
          testData = new File("testdata/points");
          if (!testData.exists()) {
            testData.mkdir();
          }
          
          // Write initial centers
          Configuration conf = new Configuration();
          
          FileSystem fs = FileSystem.get(conf);

          // Write vectors to input directory
          writePointsToFile(vectors,
              "testdata/points/file1", fs, conf);
          
          Path path = new Path("testdata/clusters/part-00000");
          
          SequenceFile.Writer writer = 
        		  new SequenceFile.Writer(fs, conf, path, Text.class, Kluster.class);
          
          for (int i = 0; i < k; i++) {
            Vector vec = (Vector) vectors.get(i);
            
            // write the initial center here as vec
            Kluster cluster = new Kluster(vec, i, new EuclideanDistanceMeasure());
            writer.append(new Text(cluster.getIdentifier()), cluster);
          }
          
          writer.close();
          
          // Run K-means algorithm
        KMeansDriver.run(conf, new Path("testdata/points"), new Path("testdata/clusters"),
        new Path("output"), 0.001, 10, true, 0.5, false);
        SequenceFile.Reader reader
              = new SequenceFile.Reader(fs,
                  new Path("output/" + Cluster.CLUSTERED_POINTS_DIR
                      + "/part-m-00000"), conf);
        IntWritable key = new IntWritable();
        
        // Read output values
        WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable(); 
        while (reader.next(key, value)) {
            System.out.println(value.toString() + " belongs to cluster "
                    + key.toString());
        }
          reader.close();
    }

}