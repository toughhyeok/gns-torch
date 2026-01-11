import os

import torch
import torch.nn.functional as F
from absl import app
from absl import flags
from absl import logging
from torch_geometric.loader import DataLoader

from dataset import GNSDataset
from graph_network import GNS

flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_epochs', 1, help='Number of epochs of training.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

FLAGS = flags.FLAGS
LR = 1e-4


def main(_):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # -----------------------------------------
    # 2. 데이터 & 모델 준비
    # -----------------------------------------
    train_dataset = GNSDataset(data_dir=os.path.join(FLAGS.data_path, 'train'), window_length=7, mode='train')

    # 로더 (PyG 로더 사용!)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=4
    )

    # 모델 초기화
    model = GNS(input_dim=2, hidden_size=128, num_layers=10, radius=0.015).to(device)

    # 최적화 도구
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------
    # 3. 학습 루프 (Training Loop)
    # -----------------------------------------
    logging.info("🚀 학습 시작!")
    model.train()

    epoch = batch_idx = 0

    for epoch in range(FLAGS.num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)

            # --- [중요] 타겟 가속도(Ground Truth Acceleration) 계산 ---
            # 데이터셋은 '다음 위치(y)'를 줍니다.
            # 하지만 우리는 '가속도'를 맞춰야 하므로 변환합니다.
            # 가속도 = 다음위치 - 현재위치 - 현재속도
            # a_t = p_{t+1} - p_t - v_t
            #     = p_{t+1} - 2*p_t + p_{t-1}

            next_pos = batch.y  # p_{t+1} (Target)
            curr_pos = batch.x[:, -1]  # p_t (Current)
            prev_pos = batch.x[:, -2]  # p_{t-1} (Previous)

            # 정답 가속도 계산
            # (주의: 노이즈가 섞인 입력 기준으로 가속도를 계산해야 모델이 노이즈 보정을 배웁니다)
            current_vel = curr_pos - prev_pos
            target_acc = next_pos - curr_pos - current_vel

            # --- Forward & Backward ---
            optimizer.zero_grad()

            # 모델 예측 (pred_acc)
            pred_acc = model(batch)

            # Loss 계산 (가속도 끼리 비교)
            loss = F.mse_loss(pred_acc, target_acc)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():e}")

            if (epoch * len(train_loader) + batch_idx) % 5000 == 0:
                torch.save(model.state_dict(),
                           os.path.join(FLAGS.model_path, f"gns_model_{epoch + 1}_{batch_idx + 1}.pth"))

        avg_loss = total_loss / num_batches
        logging.info(f"=== Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.6f} ===\n")

    logging.info("🎉 학습 완료!")

    # -----------------------------------------
    # 4. 모델 저장
    # -----------------------------------------
    torch.save(model.state_dict(), os.path.join(FLAGS.model_path, f"gns_model_{epoch + 1}_{batch_idx + 1}.pth"))
    logging.info("모델 저장 완료: gns_model_water.pth")


if __name__ == '__main__':
    app.run(main)
