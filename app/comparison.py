import numpy as np

def calculate_growth(vol1, vol2):
    """计算体积增长率"""
    if vol1 == 0:
        return float('inf') if vol2 > 0 else 0
    return ((vol2 - vol1) / vol1) * 100

def perform_comparison(report1, report2, distance_threshold_mm=10.0):
    """
    比较两次筛查报告中的结节。
    report1应为较早的报告。
    """
    nodules1 = report1.analysis_result.get('nodules', [])
    nodules2 = report2.analysis_result.get('nodules', [])

    # 如果任一报告没有结节，则无法比较
    if not nodules1 or not nodules2:
        return {
            "matched_pairs": [],
            "new_nodules": nodules2,
            "disappeared_nodules": nodules1,
            "summary": "One of the reports has no nodules, direct comparison is not possible."
        }

    # 创建一个标记来追踪哪些新结节已经被匹配
    is_nodule2_matched = [False] * len(nodules2)
    matched_pairs = []
    
    # 遍历旧报告中的每个结节
    for nodule1 in nodules1:
        best_match_nodule2 = None
        min_distance = float('inf')
        best_match_index = -1

        pos1 = np.array(nodule1['position'])

        # 在新报告中为它寻找最佳匹配
        for i, nodule2 in enumerate(nodules2):
            if not is_nodule2_matched[i]: # 只和尚未匹配的结节比较
                pos2 = np.array(nodule2['position'])
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_nodule2 = nodule2
                    best_match_index = i

        # 如果找到了在阈值内的匹配项
        if min_distance < distance_threshold_mm:
            is_nodule2_matched[best_match_index] = True
            
            # 计算变化
            volume_growth = calculate_growth(nodule1['size_mm3'], best_match_nodule2['size_mm3'])
            nodule_prob_change = best_match_nodule2['nodule_malignancy_prob'] - nodule1['nodule_malignancy_prob']
            tumor_prob_change = best_match_nodule2['tumor_malignancy_prob'] - nodule1['tumor_malignancy_prob']

            matched_pairs.append({
                'nodule1': nodule1,
                'nodule2': best_match_nodule2,
                'distance': min_distance,
                'volume_growth_percent': volume_growth,
                'nodule_prob_change': nodule_prob_change,
                'tumor_prob_change': tumor_prob_change
            })

    # 分离出未匹配的结节
    disappeared_nodules = [n for i, n in enumerate(nodules1) if n not in [p['nodule1'] for p in matched_pairs]]
    new_nodules = [n for i, n in enumerate(nodules2) if not is_nodule2_matched[i]]

    summary = (f"Found {len(matched_pairs)} matched nodule(s), "
               f"{len(new_nodules)} new nodule(s), and "
               f"{len(disappeared_nodules)} disappeared nodule(s).")

    return {
        "matched_pairs": matched_pairs,
        "new_nodules": new_nodules,
        "disappeared_nodules": disappeared_nodules,
        "summary": summary
    } 