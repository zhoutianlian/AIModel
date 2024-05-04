"""
模型评估及可视化
"""
import pandas as pd
import matplotlib.pyplot as plt
from settings import LOSS_WEIGHTS

w1, w2, w3, w4 = LOSS_WEIGHTS
w1 /= sum(LOSS_WEIGHTS)
w2 /= sum(LOSS_WEIGHTS)
w3 /= sum(LOSS_WEIGHTS)
w4 /= sum(LOSS_WEIGHTS)

def review(log_path, period: int = None):
	"""
	绘制训练集和验证集的损失函数及准确率变化曲线
	:param log_path:
	:param period:
	:return:
	"""
	train = pd.read_csv(log_path + '/train_history.csv', index_col=0)
	valid = pd.read_csv(log_path + '/valid_history.csv', index_col=0)

	if period is not None:
		train['period'] = (train.index - 1) // period
		train = train.groupby('period').mean()
		train.index = (train.index + 1) * period
	else:
		period = min(valid.index)

	univ = train * 0.75 + valid * 0.25

	end = max(train.index)
	directions = ('left', 'right', 'top', 'bottom')
	offs = dict.fromkeys(directions, False)

	plt.close()
	fig = plt.figure(figsize=(16, 9))
	fig.suptitle('Batch %d Review' % end)

	train_settings = {'label': 'Train', 'color': '#E33539'}
	valid_settings = {'label': 'Valid', 'color': '#205AA7'}
	univ_settings = {'label': 'Univ', 'color': '#F1AF00'}
	acc_yticks = []

	# 绘制损失函数
	ax = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=3, facecolor='#ECECEC')
	ax.set_title('Loss Function', loc='left')
	ax.plot(train.index, train['loss'], **train_settings)
	ax.plot(valid.index, valid['loss'], **valid_settings)
	ax.plot(univ.index, univ['loss'], **univ_settings)
	ax.set_xlim(0, end)
	ax.set_ylim(0, 30)
	ax.grid(color='#FFFFFF')
	[ax.spines[d].set_visible(False) for d in directions]
	ax.tick_params(**offs)
	ax.set_xticklabels([])
	ax.legend(loc='upper right')

	# 绘制平均准确率
	ax = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=3, facecolor='#ECECEC')
	ax.set_title('Weighted Average Accuracy', loc='left')
	y1 = univ['acc4']
	y2 = univ.eval('acc3 - acc4')
	y3 = univ.eval('acc2 - acc3')
	y4 = univ.eval('acc1 - acc2')

	ax.stackplot(univ.index, y1, y2, y3, y4,
	             colors=['#EC870E', '#F09C42', '#F5B16D', '#FACE9C'],
	             labels=['GICS4', 'GICS3', 'GICS2', 'GICS1'])

	ax.set_xlim(0, end)
	ax.set_ylim(0, 1)
	ax.grid(color='#FFFFFF')
	[ax.spines[d].set_visible(False) for d in directions]
	ax.tick_params(**offs)
	ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
	ax.legend(loc='upper left')

	# 绘制一级行业准确率
	ax = plt.subplot2grid((4, 5), (0, 3), rowspan=1, colspan=2, facecolor='#ECECEC')
	ax.set_title('GICS1 Accuracy', loc='left')
	ax.plot(train.index, train['acc1'], **train_settings)
	ax.plot(valid.index, valid['acc1'], **valid_settings)
	ax.plot(univ.index, univ['acc1'], **univ_settings)
	ax.set_xlim(0, end)
	ax.set_ylim(0, 1)
	ax.yaxis.tick_right()
	[ax.spines[d].set_visible(False) for d in directions]
	ax.tick_params(**offs)
	ax.set_xticklabels([])
	ax.set_yticklabels(acc_yticks)
	ax.grid(color='#FFFFFF')

	# 绘制二级行业准确率
	ax = plt.subplot2grid((4, 5), (1, 3), rowspan=1, colspan=2, facecolor='#ECECEC')
	ax.set_title('GICS2 Accuracy', loc='left')
	ax.plot(train.index, train['acc2'], **train_settings)
	ax.plot(valid.index, valid['acc2'], **valid_settings)
	ax.plot(univ.index, univ['acc2'], **univ_settings)
	ax.set_xlim(0, end)
	ax.set_ylim(0, 1)
	ax.yaxis.tick_right()
	[ax.spines[d].set_visible(False) for d in directions]
	ax.tick_params(**offs)
	ax.set_xticklabels([])
	ax.set_yticklabels(acc_yticks)
	ax.grid(color='#FFFFFF')

	# 绘制三级行业准确率
	ax = plt.subplot2grid((4, 5), (2, 3), rowspan=1, colspan=2, facecolor='#ECECEC')
	ax.set_title('GICS3 Accuracy', loc='left')
	ax.plot(train.index, train['acc3'], **train_settings)
	ax.plot(valid.index, valid['acc3'], **valid_settings)
	ax.plot(univ.index, univ['acc3'], **univ_settings)
	ax.set_xlim(0, end)
	ax.set_ylim(0, 1)
	ax.yaxis.tick_right()
	[ax.spines[d].set_visible(False) for d in directions]
	ax.tick_params(**offs)
	ax.set_xticklabels([])
	ax.set_yticklabels(acc_yticks)
	ax.grid(color='#FFFFFF')

	# 绘制四级行业准确率
	ax = plt.subplot2grid((4, 5), (3, 3), rowspan=1, colspan=2, facecolor='#ECECEC')
	ax.set_title('GICS4 Accuracy', loc='left')
	ax.plot(train.index, train['acc4'], **train_settings)
	ax.plot(valid.index, valid['acc4'], **valid_settings)
	ax.plot(univ.index, univ['acc4'], **univ_settings)
	ax.set_xlim(0, end)
	ax.set_ylim(0, 1)
	ax.yaxis.tick_right()
	[ax.spines[d].set_visible(False) for d in directions]
	ax.tick_params(**offs)
	ax.set_yticklabels(acc_yticks)
	ax.grid(color='#FFFFFF')

	plt.savefig(log_path + '/review.png')
	plt.close()

if __name__ == '__main__':
	review('save/c1', 50)
