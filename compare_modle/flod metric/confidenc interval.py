import pandas as pd
import scipy.stats as stats
import numpy as np

# 读取数据
file_paths = ['dsae_rf_fold_metrics.csv', 'dsmda_fold_metrics.csv', 'gpu_mda_fold_metrics.csv',
              'kgn_mda_fold_metrics.csv', 'sae_fold_metrics.csv']
model_names = ['DSAE_RF', 'DSMDA', 'GPUDMDA', 'KGNMDA', 'SAELGMDA']

data = {}
for file, name in zip(file_paths, model_names):
    data[name] = pd.read_csv(file)


# 显著性分析
def perform_ttest(metric):
    results = {}
    for model1 in model_names:
        for model2 in model_names:
            if model1 != model2:
                t_stat, p_value = stats.ttest_rel(data[model1][metric], data[model2][metric])
                results[f'{model1} vs {model2}'] = p_value
    return results


# 计算置信区间
def calculate_confidence_interval(metric, confidence=0.95):
    intervals = {}
    for model in model_names:
        values = data[model][metric]
        mean = np.mean(values)
        sem = stats.sem(values)
        margin = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        intervals[model] = (mean, mean - margin, mean + margin, margin)
    return intervals


# 执行显著性分析和计算置信区间
metrics = ['AUC', 'AUPR', 'F1', 'Accuracy']
ttest_results = {metric: perform_ttest(metric) for metric in metrics}
confidence_intervals = {metric: calculate_confidence_interval(metric) for metric in metrics}

# 打印结果
for metric in metrics:
    print(f"\nSignificance Test Results for {metric}:")
    for comparison, p_value in ttest_results[metric].items():
        print(f"{comparison}: p-value = {p_value}")

    print(f"\nConfidence Intervals for {metric}:")
    for model, (mean, lower, upper, margin) in confidence_intervals[metric].items():
        print(f"{model}: Mean = {mean:.2f} ± {margin:.2f} (95% CI: {lower:.2f}-{upper:.2f})")

# 将置信区间添加到表中
table = pd.DataFrame(columns=['Model', 'Metric', 'Mean ± Margin (95% CI)'])
rows = []
for metric in metrics:
    for model in model_names:
        mean, lower, upper, margin = confidence_intervals[metric][model]
        ci_str = f"{mean:.2f} ± {margin:.2f} (95% CI: {lower:.2f}-{upper:.2f})"
        rows.append({'Model': model, 'Metric': metric, 'Mean ± Margin (95% CI)': ci_str})

table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)

print("\nTable with Confidence Intervals:")
print(table)