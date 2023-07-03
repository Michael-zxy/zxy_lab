import numpy as np
from scipy.optimize import *
from scipy.stats import *
from scipy.integrate import quad
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib
# from empyrical import *

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['font.serif'] = ['KaiTi']
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.unicode_minus'] = False

inflation = 0.025
riskless_rate = 0.03
early_by = 0

stock_ann_return = 0.1
bond_ann_return = 0.04
stock_ann_vol = 0.20
bond_ann_vol = 0.025
correl = -0.095
mean = np.array([(1+stock_ann_return)**(1/12)-1,(1+bond_ann_return)**(1/12)-1])
cov = np.array([[stock_ann_vol**2/12,stock_ann_vol*bond_ann_vol*correl/12],[stock_ann_vol*bond_ann_vol*correl/12,bond_ann_vol**2/12]])

class employee:
	def __init__(self, wage_now, age_now, gender, theta, tgt_replacement=0.5, wage_increase_rate=0.035, age_retire=None):
		self.wage_now = wage_now
		self.tgt_replacement = tgt_replacement
		self.gender = gender
		self.age_now = age_now
		self.age_retire = age_retire
		if self.gender=='m' and self.age_retire==None:
			self.age_retire = 65
		elif self.gender=='w' and self.age_retire==None:
			self.age_retire = 60
		self.wage_increase_rate = wage_increase_rate
		self.least_tot_fund = self.least_total_fund_retire()
		self.theta = theta
		self.opt = pd.DataFrame()
		# print(self.least_tot_fund)


	def least_total_fund_retire(self):
		wage_at_retirement = self.wage_now*(1+self.wage_increase_rate)**(self.age_retire-self.age_now)
		annual_income = wage_at_retirement*12
		life_table = pd.read_excel("D:\\jx-quant\\下滑轨道\\生命表.xlsx",skiprows=[0],index_col=0)
		if self.gender == 'm':
			life_table = life_table.iloc[:,-2]
		else:
			life_table = life_table.iloc[:,-1]
		life_table = life_table.iloc[self.age_retire:]
		survive_rate = (1-life_table).cumprod()
		discount_rate = ((1+inflation)/(1+riskless_rate))**np.arange(len(survive_rate))

		return survive_rate.mul(discount_rate).sum()*annual_income

	def glide_path(self,x):
		months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
		# L0 = (stock_ann_return-bond_ann_return)/stock_ann_vol**2/self.theta
		L0 = x[2]
		L = x[3]-L0
		glide_path = L0+L*(1-1/(1+np.exp(-x[0]*((np.floor(months/12)-x[1])))))
		# plt.plot(months,glide_path)
		# plt.show()
		return glide_path

	def optimize(self):
		results = []
		for i in range(10):
			print(i)
			rand_rets = np.random.multivariate_normal(mean,cov,(1000,(self.age_retire-self.age_now)*12-early_by*12))

			# top = 0.8
			result = (minimize(self.mean_utility,[0.5,(self.age_retire-self.age_now)/2,0.2,0.7],args=(rand_rets,),bounds=[(0.1,1),((self.age_retire-self.age_now)*0.25,(self.age_retire-self.age_now)*0.75),(0,0.2),(0.6,0.8)]))
			gpth = pd.Series(self.glide_path(result.x)).iloc[::12]
			# print(result.x,self.mean_utility(result.x,rand_rets),(gpth-gpth.shift(1)).min(),\
			# 	self.mean_repl_rate(result.x,rand_rets))
			results.append(result.x)
		results = pd.DataFrame(results)

		opt = results.mean().values
		self.opt = opt
		print(opt)
		glide_path = self.glide_path(opt)
		pd.Series(glide_path).plot()
		plt.xlabel('月份')
		plt.ylabel('权益仓位')
		plt.show()
		rand_rets = np.random.multivariate_normal(mean,cov,(10000,(self.age_retire-self.age_now)*12-early_by*12))
		ret_mat = (pd.DataFrame(rand_rets[:,:,0]).mul(glide_path)+pd.DataFrame(rand_rets[:,:,1]).mul(1-glide_path))
		months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
		wages = (1+self.wage_increase_rate)**np.floor(months/12)
		output = ((((ret_mat.T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages,axis=0).sum())))
		print(output.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
		output.apply(self.irr).hist(bins=20)
		plt.xlabel('定投收益率')
		plt.ylabel('频数')
		plt.show()


		return pd.Series(self.glide_path(opt)).iloc[::12].to_frame()

	def result(self, opt):
		glide_path = self.glide_path(opt)
		pd.DataFrame(glide_path).to_excel("glide_path_中低风险.xlsx")
		rand_rets = np.random.multivariate_normal(mean,cov,(3000,(self.age_retire-self.age_now)*12-early_by*12))
		ret_mat = (pd.DataFrame(rand_rets[:,:,0]).mul(glide_path)+pd.DataFrame(rand_rets[:,:,1]).mul(1-glide_path))
		months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
		wages = (1+self.wage_increase_rate)**np.floor(months/12)
		output = ((((ret_mat.T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages,axis=0))))
		(output.sum()/wages.sum()).hist(bins = 20)
		plt.show()
		# assets = (ret_mat.T.apply(lambda x:(pd.DataFrame(np.triu(pd.DataFrame([x.tolist()]*len(x)))).T.replace(0,np.nan)+1).cumprod().mul(wages).sum(axis=1)))
		# # assets.pct_change().rolling(12).apply(max_drawdown).iloc[12:].T.describe(percentiles=[0.1,0.3,0.5,0.7,0.9]).iloc[-6:-1].T.plot()
		# # ((assets-assets.rolling(12).max()).rolling(12).min()/wages.sum()).iloc[12:].T.describe(percentiles=[0.1,0.3,0.5,0.7,0.9]).iloc[-6:-1].T.plot()
		# # plt.show()
		# print(assets.apply(lambda x:(x-x.cummax()).min()/wages.sum()).describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
		#
		# assets_return = assets.div(wages.cumsum(),axis=0)
		# print(assets_return.pct_change().apply(max_drawdown).describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
		#
		# print((assets).pct_change().apply(max_drawdown).describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
		#
		# print((pd.DataFrame(np.triu(pd.DataFrame([ret_mat.T.iloc[:,0].tolist()]*len(ret_mat.T.iloc[:,0])))).T.replace(0,np.nan)+1).cumprod().mul(wages).sum(axis=1))
		# glide_path = np.sqrt(glide_path**2*cov[0][0]+(1-glide_path)**2*cov[1][1]+2*glide_path*(1-glide_path)*cov[0][1])*np.sqrt(12)
		# pd.Series(glide_path).plot()
		# plt.show()
		# print(output.sum().apply(self.irr).describe())
		# output = output.rolling(12).sum().iloc[11::12].T.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).T
		#
		# # print(wages[::12])
		# output = output.iloc[::-1].cumsum().iloc[::-1].div(wages[::12],axis=0).iloc[:,1:]
		return output

	def mean_utility(self,x,rand_rets):
		glide_path = self.glide_path(x)
		months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
		wages = (1+self.wage_increase_rate)**np.floor(months/12)

		ret_mat = (pd.DataFrame(rand_rets[:,:,0]).mul(glide_path)+pd.DataFrame(rand_rets[:,:,1]).mul(1-glide_path))
		output = ((((ret_mat.T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages,axis=0).sum())))

		output = output/self.least_tot_fund

		utilities = output**(1-self.theta)/(1-self.theta)
		mean_utility = utilities.mean()

		return -mean_utility

	def irr(self,output):
		months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
		wages = (1+self.wage_increase_rate)**np.floor(months/12)
		return (fsolve(lambda x:np.dot(wages,(1+x)**((self.age_retire-self.age_now)*12-months))-output,0.05)[0]+1)**12-1

	def mean_repl_rate(self,x,rand_rets):
		glide_path = self.glide_path(x)
		months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
		wages = (1+self.wage_increase_rate)**np.floor(months/12)
		# print(np.sum(wages))

		ret_mat = (pd.DataFrame(rand_rets[:,:,0]).mul(glide_path)+pd.DataFrame(rand_rets[:,:,1]).mul(1-glide_path))
		# print((ret_mat.T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages,axis=0))
		output = ((((ret_mat.T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages,axis=0).sum())))

		# print(output.apply(self.irr).quantile(0.2))
		# print((output/np.sum(wages)).describe())

		# output = 0.2*output/self.least_tot_fund


		return self.least_tot_fund*0.3/output.quantile(0.2)



if __name__ == '__main__':
	# print(np.random.multivariate_normal(mean,cov,(100,2))[0])

	# tom = employee(1, 37, 'm', 5.85, wage_increase_rate = 0)
	#
	# temp_result = tom.optimize()

	self = employee(1, 37, 'm', 5.85, wage_increase_rate = 0)

	temp_result = self.optimize()
    
    rand_rets = np.random.multivariate_normal(mean,cov,(3000,(self.age_retire-self.age_now)*12-early_by*12))

	weight_ts = self.glide_path(self.opt)
    weight_ts_1 = weight_ts/weight_ts * 0.2
    weight_ts_2 = weight_ts/weight_ts * 0.5
    weight_ts_3 = weight_ts/weight_ts * 0.8

    glide_path = weight_ts
    
    ret_mat = (pd.DataFrame(rand_rets[:,:,0]).mul(glide_path)+pd.DataFrame(rand_rets[:,:,1]).mul(1-glide_path))
    months = np.arange((self.age_retire-self.age_now)*12-early_by*12)
    wages = (1+self.wage_increase_rate)**np.floor(months/12) * 0.1
    output = ((((ret_mat.T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages,axis=0))))
    output_1 = ((((ret_mat.iloc[:,0:-24].T.iloc[::-1]+1).cumprod().iloc[::-1].mul(wages[0:-24], axis=0))))
    (( output_1.sum() - output.sum()  ) > 10).sum()
    
    
    output.to_excel('D:\jx-quant\下滑轨道\测试结果.xlsx')


    temp_result.to_excel(str(self.age_now)+"_"+self.gender+"_"+str(self.theta)+".xlsx")
    
    


	# (tom.result([ 0.30571758, 10.00221566,  0.3,         1. ]))
	# tom.result([0.33815479,  13.15352553,   0.2,          0.6])
