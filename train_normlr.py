import torch
import ogb
import pandas as pd
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

from model import DualGraphormer,Unbalanced_BCE_logits_loss
from lr import PolynomialDecayLR


def get_result_dict(y_preds,y):
    return {"y_true": torch.concat(y,dim=0),
            "y_pred": torch.concat(y_preds,dim=0)}

if __name__=="__main__":
    epochs = 12
    batch_size = 128
    peak_lr = 2e-4
    end_lr = 1e-9
    total_updates = 33000 * epochs / batch_size
    warmup_updates = total_updates // 10

    torch.manual_seed(114514)

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='../../dataset/')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    print("train loader length:",len(train_loader))
    print("val loader length:", len(val_loader))
    print("test loader length:", len(test_loader))

    my_model=DualGraphormer(
        hidden_size=512,
        ffn_size=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        num_heads=32,
        num_classes=1,
        input_dropout_rate=0,
        n_layers=6,
        num_hops=5,
        dataset_name="ogbg-molhiv",
    ).cuda()

    optimizer = torch.optim.AdamW(my_model.parameters(), lr=peak_lr, betas=(0.99, 0.999))
    scheduler = PolynomialDecayLR(
        optimizer=optimizer,
        warmup_updates=warmup_updates,
        tot_updates=total_updates,
        lr=peak_lr,
        end_lr=end_lr,
        power=1
    )

    loss_func=Unbalanced_BCE_logits_loss().cuda()
    evaluator=ogb.graphproppred.Evaluator('ogbg-molhiv')

    records=[]
    for epochi in range(epochs):
        my_model.train()
        loss_train=0
        y_preds_train=[]
        y_train=[]
        train_roc=0
        for (stepi, x) in enumerate(train_loader, start=1):
            y_pred = my_model(x)
            y = x.y.to(torch.float32).cuda()
            lossi=loss_func(y_pred,y)
            lossi.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss_train=loss_train+lossi.cpu().item()
            y_train.append(y.cpu().detach())
            y_preds_train.append(y_pred.cpu().detach())
            torch.cuda.empty_cache()
            if stepi%20==0:
                print("Training-> epoch:",epochi,"step:",stepi,"loss:",loss_train/stepi,
                      "rocauc:",evaluator.eval(get_result_dict(y_preds_train,y_train)))
        #get_acc(y_preds_train, y_train)
        result_dict = evaluator.eval(get_result_dict(y_preds_train,y_train))
        print("Finish training, epoch:",epochi,result_dict)
        train_roc=evaluator.eval(get_result_dict(y_preds_train,y_train))
        print("rocauc:",train_roc)

        loss_val = 0
        y_preds_val = []
        y_val = []
        val_roc=0
        my_model.eval()
        with torch.no_grad():
            for (stepi, x) in enumerate(val_loader, start=1):
                y_pred = my_model(x)
                y = x.y.to(torch.float32).cuda()
                lossi = loss_func(y_pred, y)
                loss_val = loss_val + lossi.cpu().item()

                y_val.append(y.cpu().detach())
                y_preds_val.append(y_pred.cpu().detach())
                torch.cuda.empty_cache()
                if stepi % 20 == 0:
                    print("Validating-> epoch:", epochi, "step:", stepi, "loss:", loss_val/stepi,
                          "rocauc:",evaluator.eval(get_result_dict(y_preds_val,y_val)))

        result_dict = evaluator.eval(get_result_dict(y_preds_val, y_val))
        print("Finish validating, epoch:", epochi, result_dict)
        val_roc=evaluator.eval(get_result_dict(y_preds_val,y_val))
        print("rocauc:",val_roc)

        loss_test = 0
        y_preds_test = []
        y_test = []
        my_model.eval()
        with torch.no_grad():
            for (stepi, x) in enumerate(test_loader, start=1):
                y_pred = my_model(x)
                y = x.y.to(torch.float32).cuda()
                lossi = loss_func(y_pred, y)
                loss_test = loss_test + lossi.item()

                y_test.append(y.cpu().detach())
                y_preds_test.append(y_pred.cpu().detach())
                torch.cuda.empty_cache()
                if stepi % 20 == 0:
                    print("Testing-> epoch:", epochi, "step:", stepi, "loss:", loss_test/stepi,
                          "accuracy:","rocauc:",evaluator.eval(get_result_dict(y_preds_test,y_test)))

        result_dict = evaluator.eval(get_result_dict(y_preds_test, y_test))
        print("Finish testing, epoch:", epochi, result_dict)
        test_roc=evaluator.eval(get_result_dict(y_preds_test,y_test))
        print("rocauc:",test_roc)

        records.append([epochi,loss_train,loss_val,loss_test,train_roc,val_roc,test_roc])
        df = pd.DataFrame(columns=["epoch","loss_train","loss_val","loss_test","train_roc","val_roc","test_roc"],data=records)
        df.to_csv("./records.csv")