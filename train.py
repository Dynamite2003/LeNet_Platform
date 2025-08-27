import argparse
import random
import torch
import torchvision
import matplotlib.pyplot as plt
from models.lenet import LeNet
from utils import pre_process


def get_data_loader(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    return train_loader, test_loader


def evaluate(model, test_loader, device, criterion=None, show_errors:bool=False,max_show_num:int = 5):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.0
        error_samples = []
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 计算验证损失
            if criterion is not None:
                loss = criterion(outputs, labels)
                test_loss += loss.item()

            # 收集错误样本的image和label
            if show_errors:
                error_mask = (predicted != labels)
                for i in range(len(images)):
                    if error_mask[i]:
                        error_samples.append(
                            {
                                'image':images[i].cpu(),
                                'true_label':labels[i].cpu().item(),
                                'predicted_label':predicted[i].cpu().item()
                            }
                        )

        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader) if criterion is not None else 0.0
        
        print('Test Accuracy of the model is: {:.2f} %'.format(accuracy))
        if criterion is not None:
            print('Test Loss: {:.4f}'.format(avg_test_loss))

        # 调用show_errors现实错误样本
        if show_errors and error_samples:
            display_errors(error_samples,max_show_num)
            
        return accuracy, avg_test_loss

def display_errors(error_samples, max_show_num):
    # 显示错误样本
    num_samples = len(error_samples)

    # 从所有样本中随机选择max个（限制为5个）
    display_num = min(max_show_num, 5, num_samples)  # 最多显示5个
    if num_samples > display_num:
        random_chosen_samples = random.sample(error_samples, display_num)
    else:
        random_chosen_samples = error_samples

    # 创建static/images目录（如果不存在）
    import os
    static_dir = os.path.join('training', 'static', 'images')
    os.makedirs(static_dir, exist_ok=True)

    # 生成单个错误样本图片供网页显示
    for i, sample in enumerate(random_chosen_samples):
        image = sample['image'].squeeze().numpy()
        
        # 为每个样本创建单独的图片
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {sample["true_label"]}, Pred: {sample["predicted_label"]}', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        
        # 保存为PNG文件供网页显示
        png_path = os.path.join(static_dir, f'error_sample_{i+1}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error sample {i+1}: True={sample['true_label']}, Pred={sample['predicted_label']}")

    # 创建综合图片（原有功能）
    num_displayed = len(random_chosen_samples)
    if num_displayed > 0:
        fig, axes = plt.subplots(1, num_displayed, figsize=(2*num_displayed, 3))

        if num_displayed == 1:
            axes = [axes]

        for i, sample in enumerate(random_chosen_samples):
            image = sample['image'].squeeze().numpy()

            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'True: {sample["true_label"]}\nPred: {sample["predicted_label"]}')
            axes[i].axis('off')

        plt.suptitle(f"Error samples {num_displayed} out of total {num_samples}") 
        plt.tight_layout()
        plt.savefig("Error_samples.pdf",dpi=300)
        plt.close()
    
    return len(random_chosen_samples)  # 返回实际显示的样本数

def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def train(epochs, batch_size, learning_rate, num_classes, optimizer_type='sgd', momentum=0.9, weight_decay=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):

    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 根据优化器类型创建优化器
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
        print(f"Using Adam optimizer: lr={learning_rate}, betas=({beta1}, {beta2}), eps={eps}, weight_decay={weight_decay}")
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        print(f"Using SGD optimizer: lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")

    # start train
    total_step = len(train_loader)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):

            
            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

        # 计算epoch平均训练损失
        epoch_avg_train_loss = epoch_loss/total_step
        train_losses.append(epoch_avg_train_loss)
        print('Epoch [{}/{}] Training Loss: {:.4f}'.format(epoch + 1, epochs, epoch_avg_train_loss))
        
        # evaluate after epoch train - 获取验证损失和准确率
        val_accuracy, val_loss = evaluate(model, test_loader, device, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    # save the trained model
    save_model(model, save_path='lenet.pth')
    # plot_loss(train_losses, val_losses)

    # 最后进行一次eval 显示错误样本
    evaluate(model=model,test_loader=test_loader,device=device,criterion=criterion,show_errors=True,max_show_num=5)

    # 训练完成后导出 ONNX
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model,
        (dummy_input,),
        "lenet.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    print("模型已导出为 lenet.onnx")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    args = parser.parse_args()
    return args

# 将train的过程中的每一个epoch的loss保存下来并且绘制图像
def plot_loss(train_losses:list, val_losses:list):
    plt.figure(figsize=(10, 6))
    
    # 现在训练损失和验证损失都从第1个epoch开始，长度相同
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制训练损失
    if train_losses:
        plt.plot(epochs, train_losses, 'r-', marker='o', label='Training Loss')
        for x, y in zip(epochs, train_losses):
            plt.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=8, color='red')
    
    # 绘制验证损失（现在也从第1个epoch开始）
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'b-', marker='s', label='Validation Loss') 
        for x, y in zip(val_epochs, val_losses):
            plt.text(x, y, f'{y:.4f}', ha='center', va='top', fontsize=8, color='blue')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    train(args.epochs, args.batch_size, args.lr, args.num_classes, 
          args.optimizer, args.momentum, args.weight_decay, args.beta1, args.beta2, args.eps)


