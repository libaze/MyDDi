

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
            f"[{mode}] fold:{fold_idx} epoch:{epoch + 1} step:{step} | "
            f"loss: \033[1;31m{loss.item():.4f}\033[0m | "
            f"acc: {acc} | "
            f"f1: \033[1;32m{f1}\033[0m | "
            f"P/R: {precision}/{recall} | "
            f"auc: \033[1;32m{auc}\033[0m | "
            f"aupr: \033[1;32m{aupr}\033[0m"
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
            f"[{mode}] fold:{fold_idx} epoch:{epoch + 1} | "
            f"acc: {acc} | "
            f"f1: \033[1;32m{f1}\033[0m | "
            f"P/R: {precision}/{recall} | "
            f"auc: \033[1;32m{auc}\033[0m | "
            f"aupr: \033[1;32m{aupr}\033[0m"
        )
        # 应用到进度条
        pbar.set_description_str(pbar_desc)









