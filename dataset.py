import bisect
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class GNSDataset(Dataset):
    def __init__(self, data_dir, window_length=7, noise_std=6.7e-4, mode='train'):
        self.data_dir = data_dir
        self.window_length = window_length
        self.noise_std = noise_std
        self.mode = mode

        # 1. 파일 목록 로드
        search_path = os.path.join(data_dir, "trajectory_*.npz")
        self.file_list = sorted(glob.glob(search_path))

        # 2. 메타 데이터 생성 (각 파일에서 뽑을 수 있는 윈도우 개수 계산)
        # cumulative_lengths: [0, 994, 1988, ...] 형태로 누적된 윈도우 개수를 저장
        self.cumulative_lengths = [0]
        self.total_windows = 0

        print(f"[{mode.upper()}] Scanning {len(self.file_list)} files for indexing...")

        for f in self.file_list:
            # mmap_mode='r'로 헤더만 빠르게 읽어서 shape 확인
            try:
                with np.load(f, mmap_mode='r') as data:
                    num_steps = data['position'].shape[0]
                    # 유효한 윈도우 개수 = 전체 길이 - 윈도우 길이 + 1
                    # 예: 길이 1000, 윈도우 7 -> 0~6 ... 993~1000 -> 994개
                    valid_samples = max(0, num_steps - window_length + 1)
                    self.total_windows += valid_samples
                    self.cumulative_lengths.append(self.total_windows)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        print(f"[{mode.upper()}] Ready! Total Samples: {self.total_windows}")

    def __len__(self):
        # 전체 데이터셋의 크기는 '모든 파일에서 나올 수 있는 윈도우의 총합'
        return self.total_windows

    def __getitem__(self, idx):
        # 1. 현재 idx가 어떤 파일에 속하는지 찾기 (이진 탐색)
        # bisect_right: idx보다 큰 첫 번째 위치 반환 -> -1 해야 현재 파일 인덱스
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1

        # 2. 해당 파일 내에서의 로컬 인덱스(Start Index) 계산
        sample_idx_in_file = idx - self.cumulative_lengths[file_idx]

        # 3. 데이터 로드
        file_path = self.file_list[file_idx]

        # 실제 데이터 로드 (메모리 부족 시 mmap_mode='r' 사용 고려)
        data = np.load(file_path)

        full_pos = data['position']  # (Total_Time, N, Dim)
        particle_type = data['particle_type']  # (N,)

        # 4. 윈도우 슬라이싱 (Random 아님! 계산된 위치를 정확히 자름)
        start_t = sample_idx_in_file
        end_t = start_t + self.window_length

        pos_window = full_pos[start_t: end_t]

        # Tensor 변환
        pos_window = torch.from_numpy(pos_window).float()
        particle_type = torch.from_numpy(particle_type).long()

        # 5. Input / Target 분리
        # Input: (6, N, Dim), Target: (N, Dim) - 마지막 시점
        input_pos = pos_window[:-1]
        target_pos = pos_window[-1]

        # 6. Noise Injection (학습 모드일 때만)
        if self.mode == 'train':
            input_pos = self._add_noise(input_pos)

        # input_pos shape: (Window, N, Dim) -> GNN 입력을 위해 (N, Window, Dim)으로 변경
        x = input_pos.permute(1, 0, 2)

        data = Data(
            x=x,  # Feature: (N, 6, 2)
            y=target_pos,  # Target: (N, 2)
            pos=input_pos[-1],  # 현재 위치 (그래프 연결용): (N, 2)
            particle_type=particle_type  # 입자 타입: (N,)
        )

        return data

    def _add_noise(self, position_sequence):
        """ DeepMind 스타일 Random Walk Noise """
        velocity_sequence = position_sequence[1:] - position_sequence[:-1]
        time_steps = velocity_sequence.shape[0]

        velocity_noise = torch.randn_like(velocity_sequence)
        velocity_noise *= (self.noise_std / (time_steps ** 0.5))

        position_noise = torch.cumsum(velocity_noise, dim=0)
        zeros = torch.zeros_like(position_noise[0]).unsqueeze(0)
        position_noise = torch.cat([zeros, position_noise], dim=0)

        return position_sequence + position_noise
