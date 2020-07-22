import pandas as pd 
from simpletransformers.classification import ClassificationModel
import csv

def trainer(train_df,OUTPUT_DIR):
	model = ClassificationModel('bert', 'dbmdz/bert-base-italian-uncased', args={'train_batch_size':8, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 5, 'max_seq_length': 128,
                                                                             "fp16": False,'output_dir':OUTPUT_DIR})
	model.train_model(train_df);
	return model


def add_pred_pos (row,model,task):
  predictions, raw_outputs = model.predict([row.text_mirko])
  if(task == 'opos'):
  	return predictions[0]
  else:
  	return predictions[0]


def tester(model,task,test_df):
	if task == 'opos':
		test_df['pred_pos'] = test_df.apply(lambda row: add_pred_pos(row,model,task), axis=1)
	else:
		test_df['pred_neg'] = test_df.apply(lambda row: add_pred_pos(row,model,task), axis=1)

	test_df.to_csv(r'./res_bert_pos_new.csv',header=None,index=None, sep=',', mode='a', quotechar='"',quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
	DATA_PATH_ITA = "./data/train_new.csv"
	DATA_PATH_TEST_ITA = "./data/test_new.csv"
	OUTPUT_DIR = "./"
	task = 'oneg'

	data_train = pd.read_csv(DATA_PATH_ITA,sep=';')
	data_test = pd.read_csv(DATA_PATH_TEST_ITA,sep=";")
	if(task == 'opos'):
		OUTPUT_DIR += 'predictor_pos_3'
		train_df = data_train[['text_mirko','opos']]
		train_df.text_mirko.astype(str)
		train_df.opos.astype(float)
		test_df = data_test[['text_mirko','opos']]
		model = trainer(train_df,OUTPUT_DIR)
		#model =  ClassificationModel('bert', './predictor_pos', args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 3, 'max_seq_length': 256,
        #                                                                     "fp16": False,'output_dir':OUTPUT_DIR})
	else:
		OUTPUT_DIR += 'predictor_neg_3'
		train_df = data_train[['text_mirko','oneg']]
		train_df.text_mirko.astype(str)
		train_df.oneg.astype(float)
		test_df = data_test[['text_mirko','oneg']]
		model = trainer(train_df,OUTPUT_DIR)
		#model =  ClassificationModel('bert', './predictor_neg', args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 3, 'max_seq_length': 256,
        #                                                                     "fp16": False,'output_dir':OUTPUT_DIR})
	tester(model,task,test_df)
