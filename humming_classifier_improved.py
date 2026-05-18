"""
哼唱识别歌曲分类系统 - 改进版
==========================================

改进点：
1. 添加音高/基频(F0)特征 - 哼唱识别的核心特征
2. 添加旋律轮廓(Melody Contour)特征
3. 使用监督学习替代无监督聚类
4. 从文件名提取真实歌曲标签进行训练

数据集: MLEndHWII_sample_400
任务: 根据哼唱/口哨识别歌曲（共8首歌曲）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. 数据加载 ====================

def load_audio_files(directory_path):
    """
    加载指定目录下的所有WAV音频文件
    
    参数:
        directory_path: 音频文件目录路径
    返回:
        audio_data: 音频数据列表
        filenames: 文件名列表
        sample_rate: 采样率
    """
    audio_files = []
    file_names = []
    sr = None
    
    print("正在加载音频文件...")
    for file in os.listdir(directory_path):
        if file.endswith('.wav'):
            file_path = os.path.join(directory_path, file)
            try:
                # 加载音频，采样率22050Hz，最多10秒
                audio, sr = librosa.load(file_path, sr=22050, duration=10)
                audio_files.append(audio)
                file_names.append(file)
            except Exception as e:
                print(f"加载失败 {file}: {str(e)}")
    
    print(f"成功加载 {len(audio_files)} 个音频文件")
    return audio_files, file_names, sr


def parse_filename(filename):
    """
    从文件名解析出有用信息
    文件命名格式: S{说话者ID}_{类型}_{版本}_{歌曲名}.wav
    
    参数:
        filename: 文件名
    返回:
        speaker_id: 说话者ID
        audio_type: 音频类型 (hum/whistle)
        version: 版本号
        song_name: 歌曲名称（用于分类标签）
    """
    parts = filename.replace('.wav', '').split('_')
    
    speaker_id = parts[0]  # S100, S101等
    audio_type = parts[1]  # hum 或 whistle
    version = parts[2]     # 1, 2, 3, 4等
    song_name = parts[3]   # Happy, Feeling等歌曲名
    
    return speaker_id, audio_type, version, song_name


# ==================== 2. 特征提取（核心改进）====================

def compute_pitch_track(audio, sr, hop_length=512):
    """
    计算音高轨迹 - 核心基频提取函数（唯一调用pyin的地方）
    
    这个函数是整个特征提取流程中最耗时的操作，只需计算一次
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C1'),   # 最低频率 ~32Hz
        fmax=librosa.note_to_hz('C8'),   # 最高频率 ~4186Hz
        sr=sr,
        hop_length=hop_length
    )
    return f0, voiced_flag, voiced_probs


def extract_pitch_features_from_f0(f0, voiced_flag):
    """
    从预计算的f0提取音高特征 - 核心旋律特征
    
    参数:
        f0: 预计算的基频数组
        voiced_flag: 有声帧标记
    返回:
        pitch_features: 音高相关的统计特征 (16维)
    """
    # 处理NaN值（无音高区域）
    f0_clean = np.nan_to_num(f0, nan=0.0)
    
    pitch_features = []
    
    # 仅在有声区域计算特征
    voiced_f0 = f0_clean[voiced_flag]
    
    if len(voiced_f0) > 0:
        # 音高统计特征
        pitch_features.append(np.mean(voiced_f0))           # 平均音高
        pitch_features.append(np.std(voiced_f0))            # 音高变化
        pitch_features.append(np.median(voiced_f0))         # 中位数音高
        pitch_features.append(np.max(voiced_f0) - np.min(voiced_f0))  # 音高范围
        
        # 转换为半音计算
        voiced_f0_log = 12 * np.log2(voiced_f0 / 220)  # 相对于A4的标准差
        pitch_features.append(np.std(voiced_f0_log))   # 半音变化
        
        # 音高直方图特征（捕捉音高分布）
        pitch_hist, _ = np.histogram(voiced_f0, bins=10)
        pitch_hist = pitch_hist / (np.sum(pitch_hist) + 1e-10)  # 归一化
        pitch_features.extend(pitch_hist)
    else:
        # 无有效音高区域时填零
        pitch_features = [0] * 15
    
    # 有声比例 - 反映音频中旋律的清晰度
    voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
    pitch_features.append(voiced_ratio)
    
    return np.array(pitch_features)


def extract_melody_contour_features_from_f0(f0, voiced_flag):
    """
    从预计算的f0提取旋律轮廓特征 - 复用已有的f0结果
    
    参数:
        f0: 预计算的基频数组
        voiced_flag: 有声帧标记
    返回:
        contour_features: 旋律轮廓特征 (6维)
    """
    f0_clean = np.nan_to_num(f0, nan=0.0)
    
    # 计算一阶差分（旋律变化方向）
    f0_diff = np.diff(f0_clean)
    
    # 计算二阶差分（加速度）
    f0_diff2 = np.diff(f0_diff)
    
    contour_features = []
    
    # 旋律变化特征
    contour_features.append(np.mean(f0_diff))           # 平均变化
    contour_features.append(np.std(f0_diff))             # 变化幅度
    contour_features.append(np.mean(np.abs(f0_diff)))    # 平均绝对变化
    
    # 上升/下降比例
    if len(f0_diff) > 0:
        rising_ratio = np.sum(f0_diff > 0) / len(f0_diff)
        contour_features.append(rising_ratio)
    else:
        contour_features.append(0)
    
    # 跳跃检测（大音程变化）
    f0_diff_abs = np.abs(f0_diff)
    large_jumps = np.sum(f0_diff_abs > 200) / (len(f0_diff) + 1e-10)
    contour_features.append(large_jumps)
    
    # 音高稳定性（局部变化）
    if len(f0_clean) > 5:
        local_var = []
        for i in range(0, len(f0_clean) - 5, 5):
            local_var.append(np.std(f0_clean[i:i+5]))
        contour_features.append(np.mean(local_var))
    else:
        contour_features.append(0)
    
    return np.array(contour_features)


def extract_mfcc_features(audio, sr, n_mfcc=13, hop_length=512):
    """
    提取MFCC特征 - 保留原有特征，用于捕捉音色信息
    
    MFCC可以捕捉说话者特有的音色特征，有助于区分不同演唱风格
    """
    # 提取MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    
    # 计算统计特征
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 动态特征：一阶差分
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    
    # 二阶差分
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
    
    # 合并所有MFCC特征
    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std,      # 13 + 13 = 26
        mfcc_delta_mean, mfcc_delta_std,  # 13 + 13 = 26
        mfcc_delta2_mean           # 13
    ])
    
    return feature_vector  # 共 78 维


def extract_chroma_features(audio, sr, hop_length=512):
    """
    提取色度特征 - 捕捉音高类别信息
    
    色度特征将所有八度音折叠到同一个八度中，可以捕捉和声信息
    """
    # 计算色度特征
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
    
    # 统计特征
    chroma_mean = np.mean(chroma, axis=1)  # 12个音级的均值
    chroma_std = np.std(chroma, axis=1)   # 12个音级的变化
    
    # 色度特征向量
    chroma_features = np.concatenate([chroma_mean, chroma_std])
    
    return chroma_features  # 共 24 维


def extract_temporal_features(audio, sr):
    """
    提取时域特征 - 捕捉节奏和能量特征
    """
    features = []
    
    # 过零率 - 反映音频的粗糙程度
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # 能量
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    features.append(np.max(rms) - np.min(rms))  # 动态范围
    
    # 能量在时间上的分布（分段能量）
    n_segments = 10
    segment_size = len(audio) // n_segments
    segment_energies = []
    for i in range(n_segments):
        segment = audio[i*segment_size:(i+1)*segment_size]
        segment_energies.append(np.mean(np.abs(segment)))
    features.extend(segment_energies)  # 10个分段能量
    
    return np.array(features)  # 共 15 维


def extract_all_features(audio, sr, hop_length=512):
    """
    综合提取所有特征 - 优化版（pyin只调用一次）
    
    返回:
        完整的特征向量（多维特征组合）
    """
    # === 优化点：只计算一次pyin ===
    f0, voiced_flag, _ = compute_pitch_track(audio, sr, hop_length)
    
    # 1. 音高特征（16维）- 从预计算的f0提取
    pitch_features = extract_pitch_features_from_f0(f0, voiced_flag)
    
    # 2. 旋律轮廓特征（6维）- 从预计算的f0提取，不再重复计算pyin
    contour_features = extract_melody_contour_features_from_f0(f0, voiced_flag)
    
    # 3. MFCC特征（78维）
    mfcc_features = extract_mfcc_features(audio, sr, hop_length)
    
    # 4. 色度特征（24维）
    chroma_features = extract_chroma_features(audio, sr, hop_length)
    
    # 5. 时域特征（15维）
    temporal_features = extract_temporal_features(audio, sr)
    
    # 合并所有特征
    all_features = np.concatenate([
        pitch_features,       # 16维 - 核心
        contour_features,     # 6维
        mfcc_features,       # 78维
        chroma_features,      # 24维
        temporal_features    # 15维
    ])
    
    return all_features


def extract_features_batch(audio_data, filenames, sr):
    """
    批量提取所有音频的特征
    
    参数:
        audio_data: 音频数据列表
        filenames: 文件名列表
        sr: 采样率
    返回:
        features: 特征矩阵
        labels: 歌曲标签
        types: 音频类型标签 (hum/whistle)
    """
    features = []
    labels = []
    types = []
    
    print("\n开始特征提取...")
    for i, (audio, filename) in enumerate(zip(audio_data, filenames)):
        # 提取特征
        feat = extract_all_features(audio, sr)
        features.append(feat)
        
        # 解析标签
        _, audio_type, _, song_name = parse_filename(filename)
        labels.append(song_name)
        types.append(audio_type)
        
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(audio_data)} 个文件")
    
    features = np.array(features)
    
    print(f"\n特征提取完成!")
    print(f"特征矩阵形状: {features.shape}")
    print(f"歌曲类别: {sorted(set(labels))}")
    
    return features, labels, types


# ==================== 3. 数据预处理 ====================

def preprocess_features(features, method='standard'):
    """
    特征标准化
    
    参数:
        features: 原始特征矩阵
        method: 标准化方法 ('standard' 或 'minmax')
    返回:
        features_scaled: 标准化后的特征
        scaler: 标准化器（可用于新数据）
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, scaler


def analyze_feature_importance(features, labels, feature_names=None):
    """
    分析特征重要性（使用随机森林）
    """
    print("\n分析特征重要性...")
    
    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # 按重要性排序
    indices = np.argsort(importances)[::-1]
    
    print("\n前20个最重要的特征:")
    for i in range(min(20, len(indices))):
        print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return importances, indices


# ==================== 4. 模型训练与评估 ====================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    """
    训练多个模型并评估性能
    """
    results = {}
    
    # 定义模型
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    print("\n" + "="*60)
    print("模型训练与评估")
    print("="*60)
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"  测试集准确率: {accuracy:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"  5折交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print("\n" + "-"*60)
    print(f"最佳模型: {best_name} (准确率: {best_accuracy:.4f})")
    
    # 详细分类报告
    print("\n详细分类报告 (最佳模型):")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
    
    return results, best_model, best_name


def plot_confusion_matrix(y_true, y_pred, label_encoder, title='混淆矩阵'):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = label_encoder.classes_
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("混淆矩阵已保存为 confusion_matrix.png")


def visualize_features_pca(features, labels, label_encoder, title='特征PCA可视化'):
    """
    使用PCA降维可视化特征分布
    """
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # 编码标签
    y = label_encoder.transform(labels)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=y, cmap='tab10', alpha=0.6, s=50)
    
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)),
                label='歌曲类别')
    plt.title(title)
    plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150)
    plt.show()
    print(f"PCA可视化已保存为 pca_visualization.png")


# ==================== 5. 主程序 ====================

def main():
    """
    主函数 - 哼唱识别歌曲分类系统
    """
    print("="*60)
    print("哼唱识别歌曲分类系统 - 改进版")
    print("="*60)
    
    # 数据路径 - 请根据实际情况修改
    audio_directory = r"c:\Users\21888\Desktop\项目，哼唱识别\MLEndHWII_sample_400\MLEndHWII_sample_400"
    
    # 1. 加载音频数据
    print("\n[步骤1] 加载音频数据")
    audio_data, filenames, sample_rate = load_audio_files(audio_directory)
    
    if len(audio_data) == 0:
        print("错误: 未找到音频文件!")
        return
    
    # 2. 提取特征
    print("\n[步骤2] 提取音频特征")
    print("特征包括: 音高特征、旋律轮廓、MFCC、色度、时域特征")
    features, labels, types = extract_features_batch(audio_data, filenames, sample_rate)
    
    # 3. 预处理
    print("\n[步骤3] 特征标准化")
    features_scaled, scaler = preprocess_features(features)
    
    # 4. 编码标签
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print(f"\n歌曲类别: {list(label_encoder.classes_)}")
    print(f"类别数量: {len(label_encoder.classes_)}")
    
    # 5. 划分数据集
    print("\n[步骤4] 划分训练集和测试集")
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_encoded  # 保持类别比例
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 6. 训练和评估模型
    print("\n[步骤5] 训练和评估模型")
    results, best_model, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, label_encoder
    )
    
    # 7. 可视化
    print("\n[步骤6] 生成可视化结果")
    plot_confusion_matrix(y_test, results[best_name]['predictions'], label_encoder)
    visualize_features_pca(features_scaled, labels, label_encoder)
    
    # 8. 保存结果
    print("\n[步骤7] 保存分类结果")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': labels,
        'predicted_label': label_encoder.inverse_transform(results[best_name]['predictions']),
        'audio_type': types
    })
    
    # 添加是否正确的标记
    results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
    
    # 保存结果
    results_df.to_csv('classification_results.csv', index=False, encoding='utf-8-sig')
    print("分类结果已保存为 classification_results.csv")
    
    # 统计信息
    print("\n" + "="*60)
    print("分类统计")
    print("="*60)
    print(f"总样本数: {len(results_df)}")
    print(f"正确分类: {results_df['correct'].sum()}")
    print(f"错误分类: {(~results_df['correct']).sum()}")
    print(f"整体准确率: {results_df['correct'].mean()*100:.2f}%")
    
    # 按歌曲统计
    print("\n按歌曲分类统计:")
    for song in label_encoder.classes_:
        song_data = results_df[results_df['true_label'] == song]
        if len(song_data) > 0:
            song_acc = song_data['correct'].mean() * 100
            print(f"  {song}: {song_acc:.1f}% ({len(song_data)}个样本)")
    
    # 按类型统计 (hum vs whistle)
    print("\n按音频类型统计:")
    for audio_type in ['hum', 'whistle']:
        type_data = results_df[results_df['audio_type'] == audio_type]
        if len(type_data) > 0:
            type_acc = type_data['correct'].mean() * 100
            print(f"  {audio_type}: {type_acc:.1f}% ({len(type_data)}个样本)")
    
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    
    return best_model, scaler, label_encoder


if __name__ == "__main__":
    main()
