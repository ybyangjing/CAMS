import os
import test
import tqdm 
import torch
import numpy as np
from flags import parser
from utils import load_args,set_seed
from dataset import CompositionDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.configure_model import configure_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, dataset,args):
    model.eval()
    evaluator = test.Evaluator(dataset)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt = test.predict_logits(
            model, dataset,args)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    return test_saved_results


def train_model(model,scheduler,optimizer,args,train_dataset, val_dataset, test_dataset):
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    
    train_losses = []
    val_results = []
    
    scaler = torch.amp.GradScaler(device,enabled=True)      
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm.tqdm(
            total=len(train_loader), desc="epoch % 3d" % (epoch + 1)
        )
        epoch_train_losses = []
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.amp.autocast(device,enabled=True):
                batch = [d.to(device) if not isinstance(d, tuple) else d for d in batch]
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)   
            scaler.update()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()
        scheduler.step()
        
        progress_bar.close()
        progress_bar.write(f"epoch {epoch+1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if (epoch + 1) % args.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch_{epoch}.pt"))

        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, args)
        val_results.append(val_result)

        
        if args.val_metric == 'best_loss' and val_result[args.val_metric] < best_loss:
            best_loss = val_result['best_loss']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
            args.save_path, "val_best.pt"))
        if args.val_metric != 'best_loss' and val_result[args.val_metric] > best_metric:
            best_metric = val_result[args.val_metric]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                args.save_path, "val_best.pt"))
        
        final_model_state = model.state_dict()
        if epoch + 1 == args.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            model.load_state_dict(torch.load(os.path.join(
                args.save_path, "val_best.pt"
            )))
    evaluate(model, test_dataset, args)

    if args.save_final_model:
        torch.save(final_model_state, os.path.join(args.save_path, f'final_model.pt'))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    print(args)
    
    dataset_path = args.dataset_path
    set_seed(args.seed)
    
    train_dataset = CompositionDataset(dataset_path,
                                        phase='train',
                                        split='compositional-split-natural')

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')
    
    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')
    
    model, optimizer = configure_model(args,train_dataset)
    model.to(device)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_model(model,scheduler,optimizer,args,train_dataset, val_dataset, test_dataset) 