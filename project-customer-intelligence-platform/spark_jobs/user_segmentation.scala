import org.apache.spark.sql.SparkSession

object UserSegmentation {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("UserSegmentation").getOrCreate()
    val df = spark.read.option("header", true).csv("data/raw/users.csv")
    df.groupBy("segment").count().show()
    spark.stop()
  }
}
