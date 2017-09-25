package mllib.logisticRegression

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import java.io.Serializable
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics


object logisticRegression extends Serializable {
  
  def main(args: Array[String]): Unit = {
    
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    
    println("Process Started")
    
    val conf = new SparkConf().setAppName("Test").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlCon = new SQLContext(sc)
    
    val trainFile = sc.textFile("/home/cloudera/DataSets/kaggle/train.csv")
    
    val testFile = sc.textFile("/home/cloudera/DataSets/kaggle/test.csv")
    
    val header = trainFile.first
    
    
    val dat = trainFile.filter(x=> x != header).map(line=> line.split(",")) 
    
    val parsedData = dat.map{splitData => 
      LabeledPoint(splitData(370).toDouble,Vectors.dense(splitData.slice(0, 369).map(_.toDouble)))
    }
    
    parsedData.take(2).foreach { x => println(x) }
    
    val splits = parsedData.randomSplit(Array(0.6,0.4), seed = 11l)
    val trainingData = splits(0)
    val testData = splits(1)
    
    trainingData.cache()
    testData.cache()
    
    println("Going to Model")
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
    
    
    println("Done with Model")
    val labelAndPreds = testData.map { point =>
        val pred = model.predict(point.features)
        (point.label,pred)
    }
    
    labelAndPreds.cache()
    val trainAcc = labelAndPreds.filter(x=> x._1 == x._2).count().toDouble / testData.count()
    
    
    val multiClsMet = new MulticlassMetrics(labelAndPreds)
    val confMat = multiClsMet.confusionMatrix
    
    println("  ")
    println("Model Accuracy is   :" + trainAcc)
    println("Confusion Matrix is :" )
    println(confMat)
    println("  ")
    
    val testHeader = testFile.first
    val parsedTestData = testFile.filter( x=> x!= testHeader).map(line => line.split(","))

    println("Going to prediction")
    
    val predict = parsedTestData.map{splitData => 
        val features = Vectors.dense(splitData.slice(0, 369).map(_.toDouble))
        val pred = model.predict(features)
        (splitData(0),pred.toInt)
    }
    
    val file = predict.map{
        case(x,y) => x + "," +y
      }
    
    val csvHead = sc.parallelize(List("ID,TARGET"))

    csvHead.++(file).coalesce(1, true).saveAsTextFile("/home/cloudera/DataSets/kaggle/sample_submission.csv")
    
    println("Prediction done")
    println("Total Record Count : " + predict.count())
    println("Records with one Prediction  : " +predict.filter(x=> x._2 == 1).count)
    println("Records with zero Prediction : " +predict.filter(x=> x._2 != 1).count)
    
  }
  
}

