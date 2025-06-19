def print_log(mode, fold_idx, epoch, step, loss, metrics, pbar):
    if loss is not None and step is not None:
        # 提取指标并格式化（保留2位小数）
        acc = f"{metrics['acc']:.2f}%"
        f1 = f"{metrics['f1']:.2f}%"
        precision = f"{metrics['precision']:.2f}%"
        recall = f"{metrics['recall']:.2f}%"
        auc = f"{metrics['auc']:.2f}%"
        aupr = f"{metrics['aupr']:.2f}%"
        # 构建进度条描述
        pbar_desc = (
            f"[{mode}] fold:{fold_idx} epoch:{epoch + 1} step:{step} | loss: {loss.item():.4f} | acc: {acc} | f1: {f1} | P/R: {precision}/{recall} | auc: {auc} | aupr: {aupr}"
        )
        # 应用到进度条
        pbar.set_description_str(pbar_desc)
    else:
        # 提取指标并格式化（保留2位小数）
        acc = f"{metrics['acc']:.2f}%"
        f1 = f"{metrics['f1']:.2f}%"
        precision = f"{metrics['precision']:.2f}%"
        recall = f"{metrics['recall']:.2f}%"
        auc = f"{metrics['auc']:.2f}%"
        aupr = f"{metrics['aupr']:.2f}%"
        # 构建进度条描述
        pbar_desc = (
            f"[{mode}] fold:{fold_idx} epoch:{epoch + 1} | acc: {acc} | f1: {f1} | P/R: {precision}/{recall} | auc: {auc} | aupr: {aupr}"
        )
        # 应用到进度条
        pbar.set_description_str(pbar_desc)
