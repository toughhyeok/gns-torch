from torch import nn
from torch_geometric.nn import MessagePassing


def make_mlp(input_size, hidden_size, output_size, hidden_layers=2):
    """
    논문에서 사용하는 표준 MLP: Linear -> ReLU -> Linear ... -> LayerNorm (마지막 제외)
    """
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())

    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_size, output_size))
    # GNS는 마지막에 LayerNorm을 쓰지 않는 경우가 많지만,
    # 안정성을 위해 중간 레이어에는 넣기도 합니다. 여기선 심플하게 갑니다.
    return nn.Sequential(*layers)


import torch


def naive_radius_graph(x, r, batch=None, loop=False):
    """
    torch-cluster 라이브러리 없이 순수 PyTorch로 구현한 radius_graph.
    M1 Mac (MPS)에서도 잘 돌아갑니다.
    """
    # 배치가 없으면 모두 같은 배치로 취급
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    edge_index_list = []

    # 배치별로 따로 계산 (이렇게 해야 다른 시뮬레이션 입자와 연결 안 됨)
    # unique_batches는 보통 [0, 1] 등이 나옵니다.
    for b in torch.unique(batch):
        # 현재 배치(b)에 해당하는 입자들의 마스크(True/False)
        mask = (batch == b)

        # 현재 배치에 속한 입자들의 전역 인덱스 (Global Index)
        # 예: 2번째 배치의 입자가 전체 중 356번째~710번째라면 그 번호들
        global_indices = torch.nonzero(mask, as_tuple=True)[0]

        # 해당 배치의 입자 위치 데이터 추출
        x_batch = x[mask]

        # 거리 계산 (N_b x N_b 행렬)
        # cdist는 MPS(GPU) 가속이 잘 됩니다.
        dist_mat = torch.cdist(x_batch, x_batch)

        # 반경 r 이내인 것들 찾기 (True/False 행렬)
        adj_mask = dist_mat < r

        # 자기 자신(대각선) 제외
        if not loop:
            adj_mask.fill_diagonal_(False)

        # 연결된 인덱스 (Local Index: 0번부터 시작) 추출
        src_local, dst_local = adj_mask.nonzero(as_tuple=True)

        # Local Index -> Global Index 변환
        src_global = global_indices[src_local]
        dst_global = global_indices[dst_local]

        # 결과 저장
        edge_sub = torch.stack([src_global, dst_global], dim=0)
        edge_index_list.append(edge_sub)

    # 아무것도 연결 안 됐을 경우 예외 처리
    if len(edge_index_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    # 리스트에 모인 엣지들을 하나로 합치기
    return torch.cat(edge_index_list, dim=1)


class GNBlock(MessagePassing):
    def __init__(self, hidden_size):
        super().__init__(aggr='add')  # 이웃들의 메시지를 '더해서(add)' 받음

        # 엣지 모델: (Node + Node + Edge) -> New Edge
        # 입력: sender_node(128) + receiver_node(128) + current_edge(128) = 384
        self.edge_mlp = make_mlp(hidden_size * 3, hidden_size, hidden_size)

        # 노드 모델: (Node + Aggregated Edge) -> New Node
        # 입력: current_node(128) + aggregated_edge(128) = 256
        self.node_mlp = make_mlp(hidden_size * 2, hidden_size, hidden_size)

    def forward(self, x, edge_index, edge_attr):
        # 1. Edge Update & Message Passing
        # propagate 함수가 내부적으로 message() -> aggregate() -> update()를 호출함

        # x: [N, hidden] (노드 특징)
        # edge_index: [2, E] (연결 정보)
        # edge_attr: [E, hidden] (엣지 특징)

        # update_edge_attr을 리턴받기 위해 message 함수를 직접 호출하는 대신
        # 수동으로 계산하고 propagate를 돌립니다.

        # (src, dst) 가져오기
        src, dst = edge_index

        # 엣지 업데이트: MLP(Node_i, Node_j, Edge_ij)
        out_edge = torch.cat([x[src], x[dst], edge_attr], dim=1)
        updated_edge_attr = self.edge_mlp(out_edge)  # [E, 128]

        # 2. Node Update (Aggregation)
        # updated_edge_attr을 메시지로 보내서 노드별로 합침
        aggr_out = self.propagate(edge_index, x=x, edge_attr=updated_edge_attr, size=(x.size(0), x.size(0)))

        # 3. Residual Connection (매우 중요!)
        # 기존 엣지 특성에 잔차 연결
        new_edge_attr = edge_attr + updated_edge_attr

        # 기존 노드 특성에 잔차 연결 (propagate 결과인 aggr_out을 입력으로 사용)
        new_x = x + aggr_out

        return new_x, new_edge_attr

    def message(self, edge_attr):
        # propagate가 호출할 때 이 값을 집계(aggr='add') 함
        return edge_attr

    def update(self, aggr_out, x):
        # 집계된 메시지(aggr_out)와 내 원래 상태(x)를 합쳐서 업데이트
        input_vec = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(input_vec)


class GNS(nn.Module):
    def __init__(self, input_dim=2, hidden_size=128, num_layers=10, radius=0.015):
        super().__init__()
        self.radius = radius
        self.input_dim = input_dim  # 2D(x,y) or 3D(x,y,z)

        # ---------------------------
        # 1. ENCODER
        # ---------------------------
        # 입자 타입 임베딩 (총 9가지 타입이라 가정)
        self.type_embedding = nn.Embedding(9, 16)

        # Node Encoder:
        # 입력: 과거 5개 속도(input_dim * 5) + 벽면 거리 등(선택) + 타입임베딩(16)
        # Window가 6이면 -> 속도는 5개 나옴
        node_input_size = (input_dim * 5) + 16
        self.node_encoder = make_mlp(node_input_size, hidden_size, hidden_size)

        # Edge Encoder:
        # 입력: 상대 위치(input_dim) + 거리(1) = 3 (2D 기준)
        self.edge_encoder = make_mlp(input_dim + 1, hidden_size, hidden_size)

        # ---------------------------
        # 2. PROCESSOR
        # ---------------------------
        self.gn_layers = nn.ModuleList(
            [GNBlock(hidden_size) for _ in range(num_layers)]
        )

        # ---------------------------
        # 3. DECODER
        # ---------------------------
        # Latent(128) -> 가속도(input_dim)
        self.decoder = make_mlp(hidden_size, hidden_size, input_dim, hidden_layers=2)

    def forward(self, batch_data):
        # batch_data는 PyG DataLoader가 준 Batch 객체입니다.
        # x: (Total_N, 6, Dim) -> 위치 시퀀스
        # particle_type: (Total_N,)
        # batch: (Total_N,) -> 배치 인덱스

        pos_seq = batch_data.x  # 위치 정보
        p_type = batch_data.particle_type
        batch_idx = batch_data.batch

        # --- [전처리] 위치 시퀀스를 속도(Velocity)로 변환 ---
        # 논문에서는 절대 위치 대신 "최근 5개 속도"를 입력으로 씁니다.
        # pos_seq: (N, 6, D)
        # velocities: (N, 5, D) -> t1-t0, t2-t1, ...
        velocities = pos_seq[:, 1:] - pos_seq[:, :-1]

        # Flatten: (N, 5*D) -> MLP에 넣기 좋게 핌
        node_features = velocities.reshape(velocities.size(0), -1)

        # --- [Encoder] 1. Node Embedding ---
        # 입자 타입 임베딩
        type_emb = self.type_embedding(p_type)  # (N, 16)

        # 속도 정보 + 타입 정보 결합
        node_input = torch.cat([node_features, type_emb], dim=1)
        x = self.node_encoder(node_input)  # (N, 128) -> Latent State

        # --- [Encoder] 2. Graph Construction (Dynamic!) ---
        # 가장 최근 위치(pos_seq[:, -1])를 기준으로 반경(radius) 내 이웃 연결
        curr_pos = pos_seq[:, -1]

        # radius_graph가 배치 정보를 보고, 같은 시뮬레이션 안에서만 연결해줍니다!
        edge_index = naive_radius_graph(curr_pos, r=self.radius, batch=batch_idx, loop=False)

        # --- [Encoder] 3. Edge Embedding ---
        # sender(j) -> receiver(i)
        src, dst = edge_index

        # 상대 위치 (Relative Position): pos_j - pos_i
        rel_pos = curr_pos[src] - curr_pos[dst]
        # 거리 (Distance): ||pos_j - pos_i||
        edge_dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        # 엣지 피처: [상대위치, 거리]
        edge_input = torch.cat([rel_pos, edge_dist], dim=1)
        edge_attr = self.edge_encoder(edge_input)  # (E, 128)

        # --- [Processor] Message Passing Loop ---
        for gn_layer in self.gn_layers:
            x, edge_attr = gn_layer(x, edge_index, edge_attr)

        # --- [Decoder] Predict Acceleration ---
        pred_acc = self.decoder(x)  # (N, Dim)

        return pred_acc
