import org.apache.spark.sql.SparkSession;
import exercise_2.exercise_2;

// Launch exercise
public class UPCSchool_MLlib {
	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", "C:\\winutils");
		SparkSession spark = SparkSession.builder().master("local[*]")
				.appName("spamDetection")
				.getOrCreate();
		exercise_2.hospPrediction(spark);
		spark.close(); 
	}

}
