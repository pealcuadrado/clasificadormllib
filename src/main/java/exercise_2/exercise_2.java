package exercise_2;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class exercise_2 {
	
	
	public static Dataset<Row> readPatientsFile(SparkSession ss,
											JavaSparkContext jsc, String path)
	{
		
		//Creaamos un RDD a partir del fichero CSV con los datos de pacientes
		JavaRDD<String> fileRDD= jsc.textFile(path);
		//Filtramos para quitar el header
		fileRDD=fileRDD.filter(line-> !line.contains("\"Id\";\"IndicadorDemencia\""));
		
		//Definimos el esquema con los diferentes tipos de datos de cada columna
		StructType schema = new StructType(new StructField[] {
				DataTypes.createStructField("Id",DataTypes.StringType,true), 
				DataTypes.createStructField("IndicadorDemencia",DataTypes.StringType,true),
				DataTypes.createStructField("IndicadorConstipacion",DataTypes.StringType,true), 
				DataTypes.createStructField("IndicadorSordera",DataTypes.StringType,true),
				DataTypes.createStructField("IndicadorAltVisual",DataTypes.StringType,true), 
				DataTypes.createStructField("Barthel",DataTypes.IntegerType,true),
				DataTypes.createStructField("Pfeiffer",DataTypes.IntegerType,true), 
				DataTypes.createStructField("DiferenciaBarthel",DataTypes.IntegerType,true),
				DataTypes.createStructField("DiferenciaPfeiffer",DataTypes.IntegerType,true), 
				DataTypes.createStructField("Hemoglobina",DataTypes.DoubleType,true),
				DataTypes.createStructField("Creatinina",DataTypes.DoubleType,true), 
				DataTypes.createStructField("Albumina",DataTypes.DoubleType,true),
				DataTypes.createStructField("ListaDiagnosticosPri",DataTypes.StringType,true), 
				DataTypes.createStructField("ListaDiagnosticosSec",DataTypes.StringType,true),
				DataTypes.createStructField("ListaProcedimientosPri",DataTypes.StringType,true), 
				DataTypes.createStructField("ListaProcedimientosSec",DataTypes.StringType,true),
				DataTypes.createStructField("ListaCausasExternas",DataTypes.StringType,true), 
				DataTypes.createStructField("Reingreso",DataTypes.StringType,true),
				DataTypes.createStructField("DiasEstancia",DataTypes.IntegerType,true)
				});
		
		//Creamos el RDD<Row> desde RDD, teniendo en cuenta el tipo de datos y sustituyendo los NA por null
		JavaRDD<Row> fileRDDRow=fileRDD.map(line -> {
			Object[] attributesWithType = new Object[19];
			String[] items=line.split(";");
			attributesWithType[0]=items[0];
			attributesWithType[1]=items[1];
			attributesWithType[2]=items[2];
			attributesWithType[3]=items[3];
			attributesWithType[4]=items[4];
			attributesWithType[5]=items[5].equals("NA")? null:Integer.parseInt(items[5]);
			attributesWithType[6]=items[6].equals("NA")? null:Integer.parseInt(items[6]);
			attributesWithType[7]=items[7].equals("NA")? null:Integer.parseInt(items[7]);
			attributesWithType[8]=items[8].equals("NA")? null:Integer.parseInt(items[8]);
			attributesWithType[9]=items[9].equals("NA")? null:Double.parseDouble(items[9]);
			attributesWithType[10]=items[10].equals("NA")? null:Double.parseDouble(items[10]);
			attributesWithType[11]=items[11].equals("NA")? null:Double.parseDouble(items[11]);
			attributesWithType[12]=items[12];
			attributesWithType[13]=items[13];
			attributesWithType[14]=items[14];
			attributesWithType[15]=items[15];
			attributesWithType[16]=items[16];
			attributesWithType[17]=items[17];
			attributesWithType[18]=Integer.parseInt(items[18]);
			return RowFactory.create(attributesWithType);
		});
		
		//Finalmente, creamos el Dataset a partir del RDD y del esquema
		Dataset<Row> patientsDataset=ss.createDataFrame(fileRDDRow, schema);
		return(patientsDataset);
	}
	
	
	/*
	 * En esta función se realizan todas las transformaciones previas sobre los datos
	 * antes de la creación del modelo
	 */
	public static Dataset<Row> transformDs(Dataset<Row> ds){
		Dataset<Row> processedDs;
		
		//1.Discretizar DiasEstancias por medio de la clase Bucketizer
		double[] splits ={Double.NEGATIVE_INFINITY, 12,41,69,Double.POSITIVE_INFINITY};
			Bucketizer bucketizer = new Bucketizer()
			.setInputCol("DiasEstancia")
			.setOutputCol("DiasEstanciaCat")
			.setSplits(splits);
		processedDs = bucketizer.transform(ds);
		
		//2.Categorizar  Hemoglobina, Creatinina y Albumina. Primero sustituimos los nulos
		//y después aplicamos la clase Bucketizer
		processedDs=processedDs.na().fill(-1000l, new String[] {"Hemoglobina","Creatinina","Albumina"});
		double[][] splits2 = {
				{Double.NEGATIVE_INFINITY,-1000,12,Double.POSITIVE_INFINITY},
				{Double.NEGATIVE_INFINITY,-1000,1.11,Double.POSITIVE_INFINITY},
				{Double.NEGATIVE_INFINITY,-1000,3.5,5,Double.POSITIVE_INFINITY}
		};
		bucketizer = new Bucketizer()
				.setInputCols(new String[] {"Hemoglobina","Creatinina","Albumina"})
				.setOutputCols(new String[] {"HemoglobinaCat","CreatininaCat","AlbuminaCat"})
				.setSplitsArray(splits2);
		processedDs = bucketizer.transform(processedDs);
		
		//3. Convertir Barthel a categórica igual que en el punto anterior
		processedDs=processedDs.na().fill(-1000l, new String[] {"Barthel"});
		double[] splits3 ={Double.NEGATIVE_INFINITY, -1000,20,61,91,99,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("Barthel")
				.setOutputCol("BarthelCat")
				.setSplits(splits3);
		processedDs = bucketizer.transform(processedDs);
		
		//4. Convertir Pfeiffer a categórica igual que en el punto anterior
		processedDs=processedDs.na().fill(-1000l, new String[] {"Pfeiffer"});
		double[] splits4 ={Double.NEGATIVE_INFINITY, -1000,2,4,8,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("Pfeiffer")
				.setOutputCol("PfeifferCat")
				.setSplits(splits4);
		processedDs = bucketizer.transform(processedDs);
		
		//5. Convertir DiferenciaBarthel a categórica igual que en el punto anterior
		processedDs=processedDs.na().fill(-1000l, new String[] {"DiferenciaBarthel"});
		double[] splits5 ={Double.NEGATIVE_INFINITY, -1000,-20,20,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("DiferenciaBarthel")
				.setOutputCol("DiferenciaBarthelCat")
				.setSplits(splits5);
		processedDs = bucketizer.transform(processedDs);
		
		//6. Convertir DiferenciaPfeiffer a categórica igual que en el punto anterior
		processedDs=processedDs.na().fill(-1000l, new String[] {"DiferenciaPfeiffer"});
		double[] splits6 ={Double.NEGATIVE_INFINITY, -1000,-2,2,Double.POSITIVE_INFINITY};
		bucketizer = new Bucketizer()
				.setInputCol("DiferenciaPfeiffer")
				.setOutputCol("DiferenciaPfeifferCat")
				.setSplits(splits6);
		processedDs = bucketizer.transform(processedDs);
		
		//7. Aplicar One Hot Encoding sobre las variables anteriores
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
				  .setInputCols(new String[] {"BarthelCat", "PfeifferCat", "DiferenciaBarthelCat", "DiferenciaPfeifferCat", 
						 "HemoglobinaCat", "AlbuminaCat", "CreatininaCat"})
				  .setOutputCols(new String[] {"BarthelVect", "PfeifferVect", "DiferenciaBarthelVect", 
						  "DiferenciaPfeifferVect","HemoglobinaVect", "AlbuminaVect", "CreatininaVect"});
		OneHotEncoderModel model = encoder.fit(processedDs);
		processedDs = model.transform(processedDs);
	
		//8. Aplicar One Hot Encoding sobre el resto de variables categoricas. En este caso,
		//primero debemos indexar los Strings para que sean tratados como "levels" por medio de la 
		//clase StringIndexer
		for(String col : new String[] {"IndicadorDemencia",
						  "IndicadorConstipacion", "IndicadorSordera", "IndicadorAltVisual","Reingreso"}) {
			StringIndexer indexer = new StringIndexer();
			processedDs=indexer.setInputCol(col).setOutputCol(col+"Index").fit(processedDs).transform(processedDs);	
		}
		encoder = new OneHotEncoderEstimator()
				  .setInputCols(new String[] {"IndicadorDemenciaIndex",
						  "IndicadorConstipacionIndex", "IndicadorSorderaIndex", "IndicadorAltVisualIndex", "ReingresoIndex"})
				  .setOutputCols(new String[] {"IndicadorDemenciaVect",
						  "IndicadorConstipacionVect", "IndicadorSorderaVect", "IndicadorAltVisualVect", "ReingresoVect"});
		model = encoder.fit(processedDs);
		processedDs = model.transform(processedDs);
		
		//9. Procesado campos tipo lista. Para ello primero separamos los valor por medio de la clase
		//Tokenizer y despues vectorizamos usando Word2Vec. 
		for(String col : new String[] {"ListaDiagnosticosPri","ListaDiagnosticosSec",
				"ListaProcedimientosPri", "ListaProcedimientosSec", "ListaCausasExternas"}) {
			Tokenizer tokenizer = new Tokenizer();
			tokenizer.setInputCol(col).setOutputCol(col+"Tokenized");
			processedDs=tokenizer.transform(processedDs);
			Word2Vec word2Vec = new Word2Vec();
			word2Vec.setInputCol(col+"Tokenized").setOutputCol(col+"Vect");
			Word2VecModel modelW2V = word2Vec.fit(processedDs);
			processedDs = modelW2V.transform(processedDs);
		}
		
		return processedDs;
	}

	 
	/*
	 * En esta función se crea el modelo a partir de los datos de training.
	 * Se ha decidido implementar el clasificador usando Decision Tree.
	 * Para ello se realiza un 5-fold Cross Validation seleccionando el mejor valor posible
	 * para los hiperparámetros de maxDepth y maxBins
	 */
	public static CrossValidatorModel fitModel(Dataset<Row> train)
	{
	  
	  //Se define el modelo DecisionTree usando la columna feature que contiene todas
	  //las variables predictoras vectorizadas
	  DecisionTreeClassifier dt = new DecisionTreeClassifier()
			  .setLabelCol("DiasEstanciaCat")
			  .setFeaturesCol("features");
	  
	  //Definimos el Grid de parámetros para obtener el mejor valor para los hiperparámetros
	  //de maxDepth y maxBins
	  ParamMap[] grid = new ParamGridBuilder()
			  .addGrid(dt.maxDepth(), new int[] {5,10,25})
			  .addGrid(dt.maxBins(), new int[] {8,16,32,64,128})
			  .build();
	  
	  //Definimos el evaluator que se va a usar para seleccionar el mejor modelo
	  //En este caso, al tratarse de un clasificador para más de dos clases
	  // Seleccionamos el MulticlassClassifierEvaluator
	  MulticlassClassificationEvaluator evaluator= new MulticlassClassificationEvaluator();
	  evaluator.setLabelCol("DiasEstanciaCat").setPredictionCol("prediction");
	  
	  //Finalmente definimos el método de validación cruzada indicando que se será 5-fold CV
	  //Y se vinculan el modelo, grid y evaluator definidos anteriormente
	  CrossValidator cv = new CrossValidator()
	  .setNumFolds(5)
	  .setEstimator(dt)
	  .setEstimatorParamMaps(grid)
	  .setEvaluator(evaluator);
	  
	  CrossValidatorModel cvModel;	 
	  cvModel=cv.fit(train);
       
	  return(cvModel);
	}
	    
	
	public static void hospPrediction(SparkSession ss) {

		// Obtenemos el spark context
		JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());

		// Creacion Dataset a partir del fichero
		Dataset<Row> ds = readPatientsFile(ss,jsc,"src/main/resources/2_TR");
	    
		//Aplicamos las transformaciones al dataset anterior para tener los datos
		//en el formato deseado
		Dataset<Row> processedDs= transformDs(ds);
	    
		//Vectorizamos todas las features que se van a usar como variables predictoras.
		//Esto es necesario ya tanto el entrenamiento del modelo como la predicción usando el mismo
		//Requieren que las variables se encuentren en este formato
		 VectorAssembler assembler=new VectorAssembler();
		 assembler.setInputCols(new String[] {
				  "CreatininaVect",
				  "PfeifferVect",
				  "DiferenciaPfeifferVect",
				  "AlbuminaVect",
				  "BarthelVect",
				  "DiferenciaBarthelVect",
				  "HemoglobinaVect",
				  "IndicadorDemenciaVect",
				  "IndicadorConstipacionVect",
				  "IndicadorSorderaVect",
				  "IndicadorAltVisualVect",
				  "ListaDiagnosticosPriVect",
				  "ListaDiagnosticosSecVect",
				  "ListaProcedimientosPriVect",
				  "ListaProcedimientosSecVect",
				  "ListaCausasExternasVect",
				  "ReingresoVect"
				  })
		  .setOutputCol("features");
		  processedDs=assembler.transform(processedDs);
		  
	    // Dividimos el dataset en train y test con una proporción 20%-80%. Al tratarse de un 
		// RandomSplit estas proporciones serán aproximadas
	    Dataset<Row>[] splits= processedDs.randomSplit(new double[] {0.2,0.8},1234);
	    Dataset<Row> train = splits[1];
	    Dataset<Row> test = splits[0];	      

	    //Ajustamos el modelo definido anteriormente para los datos de training
	    CrossValidatorModel cvModel = fitModel(train);

	    //Realizamos la predicción sobre los datos de test usando el modelo que se acaba de obtener.
	    Dataset<Row> predictions=cvModel.transform(test);
 
	    //Finalmente, obtenemos las métricas de interés por medio de la clase MulticlassMetrics
	    //Ya que estamos clasificando valores de más de 2 clase (4 concretamente)
	    predictions=predictions.select("prediction","DiasEstanciaCat");
	    MulticlassMetrics mm = new MulticlassMetrics(predictions);
	    //Obtenemos el número de muestras de Train
	    System.out.println("Num muestras train: " + train.collectAsList().size() );
	    //Obtenemos el número de muestras de Test
	    System.out.println("Num muestras test: " + test.collectAsList().size() );
	    //Obtenemos el Accuracy del modelo:
	    System.out.println("Accuracy "+mm.accuracy());
	    //Obtenemos la matriz de confusión:
	    System.out.println("ConfusionMatrix");
	    System.out.println(mm.confusionMatrix().toString());
	    //Iteramos sobre las 4 categorías y obtenemos, para cada una, su precision y su recall
	    for(double label : mm.labels()) {
	    	System.out.println("LABEL (ID): "+label);
	    	System.out.println("Preccision " + mm.precision(label));
	    	System.out.println("Recall " + mm.recall(label)); 
	    }
	    
	    //Mostramos los valores óptimos de maxBins y maxDepths extraídos del mejor modelo
	    System.out.println("Mejor valor MaxBins: " + cvModel.bestModel().get(cvModel.bestModel().getParam("maxBins")).get());
	    System.out.println("Mejor valor MaxDepth: " + cvModel.bestModel().get(cvModel.bestModel().getParam("maxDepth")).get());
	    
	    ss.stop(); 
	    
	}
}
