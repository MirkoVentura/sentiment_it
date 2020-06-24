import pandas as pd
import csv
import argparse

def findPredictById(row,data_pred,task):
	val = data_pred[data_pred['idtwitter'] == row.idtwitter]
	print(val)
	if task == 'pos' :
		return val['pred_pos'].values[0]
	if task == 'neg' :
		return val['pred_neg'].values[0]
	return -1



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--gold', type=str, help="gold file for evaluate sistem", default='./data/test_gold.csv')
	parser.add_argument('--predictPos', type=str, help="output dir", default='./res_bert_try_opos.csv')
	parser.add_argument('--predictNeg', type=str, help="Path of csv for train ", default='./res_bert_try_oneg.csv')
	parser.add_argument('--runName', type=str, help="name for run ", default='001')

	args = parser.parse_args()
	gold = args.gold
	predPos = args.predictPos
	predNeg = args.predictNeg
	data_data_gold = pd.read_csv(gold,sep='\t')
	data_pred = pd.read_csv(predPos,sep='|')
	data_pred.pred_pos.astype(str)

	print(data_pred.head())
	data_pred_neg = pd.read_csv(predNeg,sep='|')
	data_pred_neg.pred_neg.astype(str)

	data_data_gold['pred_pos'] = data_data_gold.apply(lambda row: findPredictById(row,data_pred,'pos'), axis=1)
	data_data_gold['pred_neg'] = data_data_gold.apply(lambda row: findPredictById(row,data_pred_neg,'neg'), axis=1)
	print(data_data_gold.head())

	data_to_save = data_data_gold[['idtwitter','subj','pred_pos','pred_neg','iro','lpos','lneg','top']]

	data_to_save.idtwitter = data_to_save.idtwitter.astype(str)
	data_to_save.subj = data_to_save.subj.astype(str)
	data_to_save.pred_pos = data_to_save.pred_pos.astype(str)
	data_to_save.pred_neg = data_to_save.pred_neg.astype(str)
	data_to_save.iro = data_to_save.iro.astype(str)
	data_to_save.lpos = data_to_save.lpos.astype(str)
	data_to_save.lneg = data_to_save.lneg.astype(str)
	data_to_save.top = data_to_save.top.astype(str)
	data_to_save.to_csv(r'./res_bert_to_evaluete_'+args.runName+'.csv',header=None,index=None, sep=',', mode='a', quotechar='"',quoting=csv.QUOTE_NONNUMERIC)