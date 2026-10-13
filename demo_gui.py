"""
demo_gui.py

Giao diện đồ họa Pygame trực quan và chuyên nghiệp cho Bomberman FFA.
Hỗ trợ đầy đủ các chế độ demo cho buổi báo cáo/bảo vệ đồ án:

  Mode 1: 1 RL Agent vs 3 RuleBots (Benchmark Showcase - Chứng minh RL vượt trội Heuristic)
  Mode 2: 4 RL Agents FFA (Self-Play Showcase - Thể hiện cân bằng đa tác tử)
  Mode 3: Human Player vs 3 RL Agents (Người chơi đấu với 3 Deep RL Bot)
  Mode 4: Human Player vs 3 RuleBots (Người chơi đấu với 3 Heuristic Bot)

Tính năng tương tác:
  - Space: Tạm dừng / Tiếp tục (Pause/Resume)
  - N (Step): Chạy từng bước một (tiện cho thuyết trình chi tiết)
  - R: Bắt đầu lại ván mới (Reset)
  - M: Quay về Menu chính
  - 1, 2, 3, 4: Đổi tốc độ (Chậm / Chuẩn / Nhanh / Siêu tốc)
  - S: Bật/Tắt vòng bo Sudden Death (thu hẹp map sau bước 200 để ép giao tranh)
  - Điều khiển người chơi: W/A/S/D hoặc Phím mũi tên (di chuyển), SPACE hoặc ENTER (đặt bom)
"""

from __future__ import annotations

import os
import sys
import glob
import math
import numpy as np
import pygame
import torch

from bomberman_env import (
    BombermanFFAEnv,
    NUM_ACTIONS,
    ACTION_STOP,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_BOMB,
    EMPTY,
    RIGID_WALL,
    WOODEN_WALL,
    POWERUP_AMMO,
    POWERUP_RANGE,
    _MOVE_DELTA,
)
from rule_based_bot import RuleBasedAgent
from train_actor_critic import (
    ActorCriticNet,
    obs_to_tensor,
    get_action_mask,
    smart_load_weights,
)


# ───────────────────────────── PALETTE MÀU HIỆN ĐẠI ─────────────────────────────

COLOR_BG = (22, 25, 37)               # Nền chính Dark Slate
COLOR_PANEL_BG = (30, 35, 52)         # Nền thanh công cụ bên phải
COLOR_PANEL_BORDER = (50, 58, 85)     # Viền panel
COLOR_TEXT_MAIN = (240, 243, 250)     # Chữ trắng sáng
COLOR_TEXT_MUTED = (145, 155, 180)    # Chữ xám mờ
COLOR_TEXT_ACCENT = (0, 229, 255)     # Chữ nhấn neon cyan

# Ô bàn cờ
COLOR_FLOOR_A = (40, 44, 62)
COLOR_FLOOR_B = (44, 49, 69)
COLOR_FLOOR_BORDER = (35, 38, 54)

# Vật cản
COLOR_RIGID_WALL = (75, 82, 105)
COLOR_RIGID_LIGHT = (105, 114, 142)
COLOR_RIGID_DARK = (45, 50, 68)

COLOR_WOOD_WALL = (165, 110, 60)
COLOR_WOOD_LIGHT = (195, 135, 80)
COLOR_WOOD_DARK = (125, 80, 40)
COLOR_WOOD_GRAIN = (140, 90, 48)

# Bom & Lửa
COLOR_BOMB_BODY = (25, 25, 32)
COLOR_BOMB_FUSE = (255, 160, 40)
COLOR_FLAME_CORE = (255, 235, 90)
COLOR_FLAME_MID = (255, 120, 30)
COLOR_FLAME_OUTER = (235, 45, 25)

# Power-ups
COLOR_AMMO_BG = (255, 195, 40)
COLOR_RANGE_BG = (255, 65, 85)

# 4 Agent Màu sắc nhận diện riêng biệt
AGENT_COLORS = [
    (0, 229, 255),    # Agent 0: Cyan Neon
    (255, 82, 82),    # Agent 1: Coral Red
    (105, 240, 174),  # Agent 2: Emerald Green
    (255, 215, 64),   # Agent 3: Amber Gold
]

AGENT_COLOR_DARKS = [
    (0, 140, 160),
    (160, 40, 40),
    (50, 150, 100),
    (170, 140, 30),
]


# ───────────────────────────── HÀM TIỆN ÍCH MODEL ─────────────────────────────

def find_best_model_checkpoint(board_size: int = 11) -> str:
    """Tự động tìm checkpoint tốt nhất trong thư mục checkpoints."""
    candidates = [
        "checkpoints/best_phaseE_4a.pt",
        "checkpoints/final_phaseE_4a.pt",
        "checkpoints/best_phaseE.pt",
        "checkpoints/final_phaseE.pt",
        "checkpoints/best_phaseE_3a.pt",
        "checkpoints/best_phaseD.pt",
        "checkpoints/final_phaseD.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


# ───────────────────────────── QUẢN LÝ TRẬN ĐẤU (CONTROLLER) ─────────────────────────────

class GameController:
    """Điều phối môi trường, các bot RL, Rule-based bots và người chơi."""

    def __init__(self, mode: int = 1, board_size: int = 11, sudden_death: bool = True):
        self.mode = mode
        self.board_size = board_size
        self.sudden_death = sudden_death
        self.max_steps = 350

        # Khởi tạo môi trường
        self.env = BombermanFFAEnv(
            board_size=board_size,
            num_agents=4,
            max_steps=self.max_steps,
            sudden_death=sudden_death,
        )
        self.obs_list = self.env.reset()

        # Tải mô hình RL Actor-Critic
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rl_net = ActorCriticNet(board_size=board_size).to(self.device)
        ckpt_path = find_best_model_checkpoint(board_size)
        if ckpt_path and os.path.exists(ckpt_path):
            smart_load_weights(self.rl_net, ckpt_path, self.device)
            self.rl_net.eval()
            self.model_status = f"Loaded {os.path.basename(ckpt_path)}"
        else:
            self.model_status = "Using Untrained Model"

        # Khởi tạo Rule-based agents
        self.rule_bots = {i: RuleBasedAgent(i) for i in range(4)}

        # Cấu hình danh tính 4 Agent theo Mode
        # Mode 1: A0 = RL, A1..3 = RULE
        # Mode 2: A0..3 = RL
        # Mode 3: A0 = HUMAN, A1..3 = RL
        # Mode 4: A0 = HUMAN, A1..3 = RULE
        self.agent_types = {}
        self.setup_agents_for_mode()

        # Trạng thái vòng lặp
        self.step_count = 0
        self.paused = False
        self.speed_fps = 8  # Tốc độ bước
        self.game_over = False
        self.winner_str = "Fighting..."
        self.human_action_buffered = ACTION_STOP

        # Nhật ký sự kiện (Event log)
        self.event_log: list[str] = ["Trận đấu mới bắt đầu!"]
        self.stats = {
            i: {"bombs": 0, "walls": 0, "kills": 0, "alive": True} for i in range(4)
        }

    def setup_agents_for_mode(self):
        if self.mode == 1:
            self.agent_types = {0: "RL", 1: "RULE", 2: "RULE", 3: "RULE"}
            self.mode_name = "1 RL Agent (Cyan) vs 3 RuleBots"
        elif self.mode == 2:
            self.agent_types = {0: "RL", 1: "RL", 2: "RL", 3: "RL"}
            self.mode_name = "4 Deep RL Agents FFA (Self-Play)"
        elif self.mode == 3:
            self.agent_types = {0: "HUMAN", 1: "RL", 2: "RL", 3: "RL"}
            self.mode_name = "Human Player vs 3 RL Agents"
        elif self.mode == 4:
            self.agent_types = {0: "HUMAN", 1: "RULE", 2: "RULE", 3: "RULE"}
            self.mode_name = "Human Player vs 3 RuleBots"

    def reset_match(self):
        self.obs_list = self.env.reset()
        self.step_count = 0
        self.game_over = False
        self.winner_str = "Fighting..."
        self.human_action_buffered = ACTION_STOP
        self.event_log = ["Ván đấu mới đã sẵn sàng!"]
        self.stats = {
            i: {"bombs": 0, "walls": 0, "kills": 0, "alive": True} for i in range(4)
        }

    def apply_sudden_death(self):
        """Keep the environment's shared sudden-death setting in sync with the UI."""
        self.env.sudden_death = self.sudden_death

    def step(self):
        if self.game_over:
            return

        self.apply_sudden_death()
        actions = [ACTION_STOP] * 4

        # Thu thập action cho từng agent
        for i in range(4):
            agent = self.env.agents[i]
            if not agent.alive:
                actions[i] = ACTION_STOP
                continue

            atype = self.agent_types[i]
            if atype == "HUMAN":
                # Lấy hành động người chơi đã bấm
                actions[i] = self.human_action_buffered
                self.human_action_buffered = ACTION_STOP  # Reset sau khi dùng
            elif atype == "RL":
                # Mạng Actor-Critic + Action Masking
                obs_t = obs_to_tensor(self.obs_list[i], self.device)
                with torch.no_grad():
                    logits, _ = self.rl_net(obs_t)
                    mask = get_action_mask(self.env, i)
                    mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
                    m_logits = logits.masked_fill(~mask_t, -1e9)
                    act = torch.distributions.Categorical(logits=m_logits).sample().item()
                    actions[i] = act
            elif atype == "RULE":
                # Rule-Based Heuristic Bot
                actions[i] = self.rule_bots[i].select_action(self.env)

        # Cập nhật số bom đặt
        for i, a in enumerate(actions):
            if a == ACTION_BOMB and self.env.agents[i].alive:
                self.stats[i]["bombs"] += 1
                if len(self.event_log) < 50:
                    self.event_log.insert(0, f"Bước {self.step_count}: Agent {i} ({self.agent_types[i]}) đặt bom!")

        # Bước tiếp môi trường
        self.obs_list, rewards, dones, info = self.env.step(actions)
        self.step_count = info["step"]

        # Thống kê tường bị phá
        wd = info.get("walls_destroyed", {})
        for owner, cnt in wd.items():
            if owner in self.stats:
                self.stats[owner]["walls"] += cnt

        # Thống kê kills & ai bị hạ gục
        for killer, victim in info.get("kills", []):
            if killer != victim and killer in self.stats:
                self.stats[killer]["kills"] += 1
                self.event_log.insert(0, f"Bước {self.step_count}: Agent {killer} hạ gục Agent {victim}!")
            elif killer == victim:
                self.event_log.insert(0, f"Bước {self.step_count}: Agent {victim} tự nổ trúng mình!")

        # Cập nhật sống sót
        for i, agent in enumerate(self.env.agents):
            self.stats[i]["alive"] = agent.alive

        # Kiểm tra kết thúc trận
        if info["episode_done"]:
            self.game_over = True
            survivors = [i for i, a in enumerate(info["alive"]) if a]
            if len(survivors) == 1:
                win_id = survivors[0]
                self.winner_str = f"AGENT {win_id} ({self.agent_types[win_id]}) CHIẾN THẮNG!"
                self.event_log.insert(0, f"🏆 {self.winner_str}")
            elif len(survivors) == 0:
                self.winner_str = "HÒA (Tất cả đã bị tiêu diệt!)"
                self.event_log.insert(0, "Hòa: Cả 4 agent cùng bị hạ gục!")
            else:
                self.winner_str = f"HÒA (Hết giờ - Còn {len(survivors)} agent sống sót)"
                self.event_log.insert(0, f"Hòa: Hết thời gian ({self.max_steps} bước)!")


# ───────────────────────────── GIAO DIỆN PYGAME (RENDERER) ─────────────────────────────

class PygameApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Bomberman FFA — Deep RL Showcase (Actor-Critic vs RuleBots)")

        # Kích thước cửa sổ
        self.screen_width = 1180
        self.screen_height = 760
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Khởi tạo font chữ
        self.font_large = pygame.font.SysFont(["Segoe UI", "Arial", "sans-serif"], 28, bold=True)
        self.font_mid = pygame.font.SysFont(["Segoe UI", "Arial", "sans-serif"], 20, bold=True)
        self.font_normal = pygame.font.SysFont(["Segoe UI", "Arial", "sans-serif"], 16)
        self.font_small = pygame.font.SysFont(["Segoe UI", "Arial", "sans-serif"], 13)

        # Trạng thái ứng dụng: "MENU" hoặc "GAME"
        self.app_state = "MENU"
        self.controller = GameController(mode=1, board_size=11)

        # Tính toán hình học bàn cờ
        self.board_offset_x = 30
        self.board_offset_y = 60
        self.board_draw_size = 660  # Pixel vuông cho bàn cờ
        self.cell_size = self.board_draw_size // self.controller.board_size

        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            if self.app_state == "GAME":
                if not self.controller.paused and not self.controller.game_over:
                    self.controller.step()
                self.render_game()
            elif self.app_state == "MENU":
                self.render_menu()

            pygame.display.flip()
            # Điều khiển FPS theo tốc độ chọn
            fps = self.controller.speed_fps if self.app_state == "GAME" else 30
            self.clock.tick(fps)

        pygame.quit()

    # ─────────────────────── XỬ LÝ SỰ KIỆN PHÍM BẤM ───────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.app_state == "GAME":
                        self.app_state = "MENU"
                    else:
                        self.running = False
                    return

                if self.app_state == "MENU":
                    # Chọn Mode từ Menu
                    if event.key == pygame.K_1:
                        self.start_game(mode=1)
                    elif event.key == pygame.K_2:
                        self.start_game(mode=2)
                    elif event.key == pygame.K_3:
                        self.start_game(mode=3)
                    elif event.key == pygame.K_4:
                        self.start_game(mode=4)
                    elif event.key == pygame.K_s:
                        self.controller.sudden_death = not self.controller.sudden_death
                    elif event.key == pygame.K_b:
                        # Đổi kích thước bàn 9 <-> 11
                        new_s = 9 if self.controller.board_size == 11 else 11
                        self.controller = GameController(mode=self.controller.mode, board_size=new_s)
                        self.cell_size = self.board_draw_size // self.controller.board_size

                elif self.app_state == "GAME":
                    # Phím điều khiển chung
                    if event.key == pygame.K_m:
                        self.app_state = "MENU"
                    elif event.key == pygame.K_r:
                        self.controller.reset_match()
                    elif event.key == pygame.K_SPACE and self.controller.agent_types[0] != "HUMAN":
                        # Space là Pause khi không có người chơi
                        self.controller.paused = not self.controller.paused
                    elif event.key == pygame.K_n and self.controller.paused:
                        # Bước tiếp 1 bước
                        self.controller.step()
                    elif event.key == pygame.K_s:
                        self.controller.sudden_death = not self.controller.sudden_death
                    # Phím đổi tốc độ
                    elif event.key == pygame.K_1:
                        self.controller.speed_fps = 3   # Chậm
                    elif event.key == pygame.K_2:
                        self.controller.speed_fps = 8   # Chuẩn
                    elif event.key == pygame.K_3:
                        self.controller.speed_fps = 18  # Nhanh
                    elif event.key == pygame.K_4:
                        self.controller.speed_fps = 40  # Siêu tốc

                    # Phím điều khiển người chơi (khi Agent 0 là HUMAN)
                    if self.controller.agent_types[0] == "HUMAN":
                        if event.key in (pygame.K_w, pygame.K_UP):
                            self.controller.human_action_buffered = ACTION_UP
                        elif event.key in (pygame.K_s, pygame.K_DOWN):
                            self.controller.human_action_buffered = ACTION_DOWN
                        elif event.key in (pygame.K_a, pygame.K_LEFT):
                            self.controller.human_action_buffered = ACTION_LEFT
                        elif event.key in (pygame.K_d, pygame.K_RIGHT):
                            self.controller.human_action_buffered = ACTION_RIGHT
                        elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_e):
                            self.controller.human_action_buffered = ACTION_BOMB
                        elif event.key == pygame.K_p:
                            self.controller.paused = not self.controller.paused

            # Xử lý chuột click trong menu
            if self.app_state == "MENU" and event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for btn in self.menu_buttons:
                    if btn["rect"].collidepoint(mx, my):
                        btn["action"]()

    def start_game(self, mode: int):
        self.controller = GameController(mode=mode, board_size=self.controller.board_size,
                                         sudden_death=self.controller.sudden_death)
        self.cell_size = self.board_draw_size // self.controller.board_size
        self.app_state = "GAME"

    # ─────────────────────── VẼ MÀN HÌNH MENU CHÍNH ───────────────────────

    def render_menu(self):
        self.screen.fill(COLOR_BG)

        # Header Title
        title_surf = self.font_large.render("BOMBERMAN MULTI-AGENT RL SHOWCASE", True, COLOR_TEXT_ACCENT)
        sub_surf = self.font_normal.render("Actor-Critic Deep RL (CNN) vs Rule-Based BFS Bots & Human Play", True, COLOR_TEXT_MUTED)
        self.screen.blit(title_surf, (self.screen_width // 2 - title_surf.get_width() // 2, 60))
        self.screen.blit(sub_surf, (self.screen_width // 2 - sub_surf.get_width() // 2, 105))

        # Model badge
        model_txt = f"Active Model: {self.controller.model_status} | Board Size: {self.controller.board_size}x{self.controller.board_size}"
        model_surf = self.font_small.render(model_txt, True, (120, 220, 160))
        self.screen.blit(model_surf, (self.screen_width // 2 - model_surf.get_width() // 2, 135))

        # 4 Khối nút Mode
        self.menu_buttons = []
        cards = [
            {
                "num": "1",
                "title": "1 RL Agent vs 3 RuleBots (Benchmark Showcase)",
                "desc": "Chứng minh Deep RL outplay và vượt trội hoàn toàn so với bot Heuristic truyền thống.",
                "color": (0, 229, 255),
                "mode": 1,
            },
            {
                "num": "2",
                "title": "4 Deep RL Agents FFA (Self-Play Showcase)",
                "desc": "Đấu trí cân tài cân sức giữa 4 bản sao của mạng nơ-ron Actor-Critic trên bàn cờ 11x11.",
                "color": (255, 82, 82),
                "mode": 2,
            },
            {
                "num": "3",
                "title": "Human Player vs 3 RL Agents (Playable Mode)",
                "desc": "Trực tiếp cầm phím WASD / Mũi tên + Space so tài với 3 bot Deep RL do bạn huấn luyện!",
                "color": (105, 240, 174),
                "mode": 3,
            },
            {
                "num": "4",
                "title": "Human Player vs 3 RuleBots (Practice Mode)",
                "desc": "Thử sức với 3 bot luật cứng cơ bản (tìm đường an toàn bằng BFS) để khởi động.",
                "color": (255, 215, 64),
                "mode": 4,
            },
        ]

        start_y = 180
        card_w = 780
        card_h = 95
        card_x = (self.screen_width - card_w) // 2

        for i, card in enumerate(cards):
            cy = start_y + i * 115
            rect = pygame.Rect(card_x, cy, card_w, card_h)

            # Hover highlight
            mx, my = pygame.mouse.get_pos()
            hover = rect.collidepoint(mx, my)
            bg_col = (42, 48, 70) if hover else COLOR_PANEL_BG
            border_col = card["color"] if hover else COLOR_PANEL_BORDER

            # Vẽ khung card
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=10)
            pygame.draw.rect(self.screen, border_col, rect, width=2 if hover else 1, border_radius=10)

            # Badge số phím tắt
            num_rect = pygame.Rect(card_x + 18, cy + 22, 50, 50)
            pygame.draw.rect(self.screen, card["color"], num_rect, border_radius=8)
            num_text = self.font_large.render(card["num"], True, (20, 24, 34))
            self.screen.blit(num_text, (num_rect.centerx - num_text.get_width() // 2,
                                        num_rect.centery - num_text.get_height() // 2))

            # Tiêu đề & mô tả
            t_surf = self.font_mid.render(card["title"], True, COLOR_TEXT_MAIN)
            d_surf = self.font_normal.render(card["desc"], True, COLOR_TEXT_MUTED)
            self.screen.blit(t_surf, (card_x + 85, cy + 18))
            self.screen.blit(d_surf, (card_x + 85, cy + 50))

            m_val = card["mode"]
            self.menu_buttons.append({"rect": rect, "action": lambda m=m_val: self.start_game(m)})

        # Footer cài đặt nhanh
        footer_y = start_y + len(cards) * 115 + 15
        sd_status = "[BẬT]" if self.controller.sudden_death else "[TẮT]"
        sd_col = (105, 240, 174) if self.controller.sudden_death else (255, 82, 82)
        help_text = f"Phím [1-4]: Chọn chế độ | [S]: Vòng bo Sudden Death {sd_status} | [B]: Đổi bàn ({self.controller.board_size}x{self.controller.board_size}) | [ESC]: Thoát"
        help_surf = self.font_normal.render(help_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(help_surf, (self.screen_width // 2 - help_surf.get_width() // 2, footer_y))

    # ─────────────────────── VẼ MÀN HÌNH GAMEPLAY ───────────────────────

    def render_game(self):
        self.screen.fill(COLOR_BG)

        # Vẽ bàn cờ game
        self.render_board()

        # Vẽ thanh Dashboard bên phải
        self.render_dashboard()

    def render_board(self):
        env = self.controller.env
        s = self.controller.board_size
        cs = self.cell_size
        ox = self.board_offset_x
        oy = self.board_offset_y

        # Khung viền ngoài bàn cờ
        board_rect = pygame.Rect(ox - 6, oy - 6, s * cs + 12, s * cs + 12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, board_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, board_rect, width=2, border_radius=12)

        # 1. Vẽ các ô sàn (Floor)
        for r in range(s):
            for c in range(s):
                x = ox + c * cs
                y = oy + r * cs
                col = COLOR_FLOOR_A if (r + c) % 2 == 0 else COLOR_FLOOR_B
                cell_rect = pygame.Rect(x, y, cs, cs)
                pygame.draw.rect(self.screen, col, cell_rect)
                pygame.draw.rect(self.screen, COLOR_FLOOR_BORDER, cell_rect, width=1)

        # 2. Vẽ Tường cứng (Rigid Walls) & Tường gỗ (Wooden Walls)
        for r in range(s):
            for c in range(s):
                val = env.grid[r, c]
                x = ox + c * cs
                y = oy + r * cs
                rect = pygame.Rect(x + 2, y + 2, cs - 4, cs - 4)

                if val == RIGID_WALL:
                    # Khối đá 3D kim loại
                    pygame.draw.rect(self.screen, COLOR_RIGID_WALL, rect, border_radius=4)
                    pygame.draw.line(self.screen, COLOR_RIGID_LIGHT, (rect.left, rect.top), (rect.right, rect.top), 2)
                    pygame.draw.line(self.screen, COLOR_RIGID_LIGHT, (rect.left, rect.top), (rect.left, rect.bottom), 2)
                    pygame.draw.line(self.screen, COLOR_RIGID_DARK, (rect.right, rect.top), (rect.right, rect.bottom), 2)
                    pygame.draw.line(self.screen, COLOR_RIGID_DARK, (rect.left, rect.bottom), (rect.right, rect.bottom), 2)
                    # Biểu tượng dấu thập kim loại
                    cx, cy = rect.center
                    pygame.draw.circle(self.screen, COLOR_RIGID_DARK, (cx, cy), 3)

                elif val == WOODEN_WALL:
                    # Khối gỗ có vân gạch
                    pygame.draw.rect(self.screen, COLOR_WOOD_WALL, rect, border_radius=4)
                    pygame.draw.line(self.screen, COLOR_WOOD_LIGHT, (rect.left, rect.top), (rect.right, rect.top), 2)
                    pygame.draw.line(self.screen, COLOR_WOOD_DARK, (rect.left, rect.bottom), (rect.right, rect.bottom), 2)
                    # Các đường vân gỗ
                    h = rect.height
                    pygame.draw.line(self.screen, COLOR_WOOD_GRAIN, (rect.left + 3, rect.top + h // 3), (rect.right - 3, rect.top + h // 3), 1)
                    pygame.draw.line(self.screen, COLOR_WOOD_GRAIN, (rect.left + 3, rect.top + 2 * h // 3), (rect.right - 3, rect.top + 2 * h // 3), 1)

        # 3. Vẽ Power-ups (Vật phẩm nhặt được)
        for (r, c), kind in env.powerups.items():
            x = ox + c * cs + cs // 2
            y = oy + r * cs + cs // 2
            rad = cs // 3
            if kind == POWERUP_AMMO:
                pygame.draw.circle(self.screen, COLOR_AMMO_BG, (x, y), rad)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), rad, 1)
                t = self.font_small.render("+B", True, (30, 30, 30))
                self.screen.blit(t, (x - t.get_width() // 2, y - t.get_height() // 2))
            else:
                pygame.draw.circle(self.screen, COLOR_RANGE_BG, (x, y), rad)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), rad, 1)
                t = self.font_small.render("+R", True, (255, 255, 255))
                self.screen.blit(t, (x - t.get_width() // 2, y - t.get_height() // 2))

        # 4. Vẽ Bom & Bộ đếm ngược thời gian (Bomb Countdown Timer)
        now_ticks = pygame.time.get_ticks()
        for b in env.bombs:
            x = ox + b.col * cs + cs // 2
            y = oy + b.row * cs + cs // 2
            # Hiệu ứng nhịp đập kích thước bom theo thời gian đếm ngược
            pulse = math.sin(now_ticks * 0.012) * 1.5 if b.timer <= 3 else 0
            rad = int(cs // 2.5 + pulse)

            # Thân quả bom tròn màu đen
            pygame.draw.circle(self.screen, COLOR_BOMB_BODY, (x, y), rad)
            pygame.draw.circle(self.screen, COLOR_FLAME_MID, (x, y), rad, 2)
            # Ngòi nổ trên đỉnh
            pygame.draw.line(self.screen, COLOR_BOMB_FUSE, (x, y - rad), (x + 3, y - rad - 5), 2)
            pygame.draw.circle(self.screen, COLOR_FLAME_CORE, (x + 3, y - rad - 5), 2)

            # Hiển thị số giây/tick đếm ngược ngay trên quả bom
            timer_col = (255, 80, 80) if b.timer <= 2 else COLOR_TEXT_MAIN
            t_surf = self.font_normal.render(str(b.timer), True, timer_col)
            self.screen.blit(t_surf, (x - t_surf.get_width() // 2, y - t_surf.get_height() // 2))

        # 5. Vẽ Ngọn Lửa Bùng Nổ (Flames)
        for (r, c), dur in env.flames.items():
            fx = ox + c * cs
            fy = oy + r * cs
            rect = pygame.Rect(fx + 1, fy + 1, cs - 2, cs - 2)

            # Lớp lửa ngoài
            pygame.draw.rect(self.screen, COLOR_FLAME_OUTER, rect, border_radius=4)
            # Lớp lửa giữa
            inner_rect = rect.inflate(-cs // 4, -cs // 4)
            pygame.draw.rect(self.screen, COLOR_FLAME_MID, inner_rect, border_radius=3)
            # Lõi lửa sáng vàng
            core_rect = rect.inflate(-cs // 2, -cs // 2)
            pygame.draw.rect(self.screen, COLOR_FLAME_CORE, core_rect, border_radius=2)

        # 6. Vẽ các Agents (Nhân vật)
        for agent in env.agents:
            if not agent.alive:
                continue

            i = agent.agent_id
            cx = ox + agent.col * cs + cs // 2
            cy = oy + agent.row * cs + cs // 2
            rad = int(cs // 2.4)

            main_col = AGENT_COLORS[i]
            dark_col = AGENT_COLOR_DARKS[i]

            # Vòng hào quang phát sáng
            pygame.draw.circle(self.screen, main_col, (cx, cy), rad + 3, 2)
            # Thân agent
            pygame.draw.circle(self.screen, main_col, (cx, cy), rad)
            pygame.draw.circle(self.screen, dark_col, (cx, cy), rad, 2)

            # Hai mắt dễ thương hướng nhìn
            eye_col = (20, 24, 34)
            pygame.draw.circle(self.screen, eye_col, (cx - 4, cy - 3), 3)
            pygame.draw.circle(self.screen, eye_col, (cx + 4, cy - 3), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx - 3, cy - 4), 1)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx + 5, cy - 4), 1)

            # ID Huy hiệu nhỏ góc dưới
            id_surf = self.font_small.render(str(i), True, (20, 24, 34))
            self.screen.blit(id_surf, (cx - id_surf.get_width() // 2, cy + 4))

    # ─────────────────────── VẼ THANH DASHBOARD BÊN PHẢI ───────────────────────

    def render_dashboard(self):
        dx = self.board_offset_x + self.board_draw_size + 25
        dy = self.board_offset_y - 6
        dw = self.screen_width - dx - 25
        dh = self.board_draw_size + 12

        panel_rect = pygame.Rect(dx, dy, dw, dh)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=2, border_radius=12)

        px = dx + 18
        py = dy + 18

        # 1. Tiêu đề Mode
        mode_surf = self.font_mid.render(self.controller.mode_name, True, COLOR_TEXT_ACCENT)
        self.screen.blit(mode_surf, (px, py))
        py += 30

        # Trạng thái trận đấu (Status Banner)
        banner_rect = pygame.Rect(px, py, dw - 36, 42)
        if self.controller.game_over:
            b_bg = (55, 30, 40)
            b_text_col = (255, 100, 100) if "HÒA" in self.controller.winner_str else (100, 255, 160)
        else:
            b_bg = (38, 44, 66)
            b_text_col = (255, 215, 64) if self.controller.paused else COLOR_TEXT_MAIN

        pygame.draw.rect(self.screen, b_bg, banner_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, banner_rect, width=1, border_radius=6)

        status_msg = f"TẠM DỪNG (Phím Space)" if (self.controller.paused and not self.controller.game_over) else self.controller.winner_str
        s_surf = self.font_normal.render(status_msg, True, b_text_col)
        self.screen.blit(s_surf, (banner_rect.centerx - s_surf.get_width() // 2,
                                  banner_rect.centery - s_surf.get_height() // 2))
        py += 52

        # Tiến trình bước (Step progress)
        step_txt = f"Bước: {self.controller.step_count} / {self.controller.max_steps}"
        fps_names = {3: "1x (Chậm)", 8: "2x (Chuẩn)", 18: "3x (Nhanh)", 40: "4x (Turbo)"}
        speed_txt = f"Tốc độ: {fps_names.get(self.controller.speed_fps, 'Chuẩn')}"
        t_surf1 = self.font_normal.render(step_txt, True, COLOR_TEXT_MAIN)
        t_surf2 = self.font_small.render(speed_txt, True, COLOR_TEXT_MUTED)
        self.screen.blit(t_surf1, (px, py))
        self.screen.blit(t_surf2, (dx + dw - 18 - t_surf2.get_width(), py + 2))
        py += 22

        # Thanh tiến trình bước (Progress Bar)
        prog_w = dw - 36
        bar_rect = pygame.Rect(px, py, prog_w, 6)
        pygame.draw.rect(self.screen, (45, 50, 70), bar_rect, border_radius=3)
        filled_w = int(prog_w * min(1.0, self.controller.step_count / self.controller.max_steps))
        if filled_w > 0:
            fill_rect = pygame.Rect(px, py, filled_w, 6)
            pygame.draw.rect(self.screen, COLOR_TEXT_ACCENT, fill_rect, border_radius=3)
        py += 18

        # 2. Bảng trạng thái 4 Agents (Agent Cards)
        for i in range(4):
            agent = self.controller.env.agents[i]
            st = self.controller.stats[i]
            card_h = 56
            card_rect = pygame.Rect(px, py, dw - 36, card_h)

            card_bg = (36, 42, 62) if st["alive"] else (28, 30, 40)
            pygame.draw.rect(self.screen, card_bg, card_rect, border_radius=6)
            pygame.draw.rect(self.screen, AGENT_COLORS[i] if st["alive"] else (60, 65, 80), card_rect, width=1, border_radius=6)

            # Avatar tròn của agent
            av_x = px + 18
            av_y = py + card_h // 2
            col = AGENT_COLORS[i] if st["alive"] else (80, 85, 100)
            pygame.draw.circle(self.screen, col, (av_x, av_y), 13)
            av_id = self.font_small.render(str(i), True, (20, 24, 34))
            self.screen.blit(av_id, (av_x - av_id.get_width() // 2, av_y - av_id.get_height() // 2))

            # Thông tin tên & role
            atype = self.controller.agent_types[i]
            role_map = {"RL": "Actor-Critic RL", "RULE": "Rule-Based Bot", "HUMAN": "BẠN (Player 1)"}
            name_txt = f"Agent {i} [{role_map[atype]}]"
            name_surf = self.font_normal.render(name_txt, True, AGENT_COLORS[i] if st["alive"] else (100, 105, 120))
            self.screen.blit(name_surf, (av_x + 20, py + 8))

            # Chỉ số chi tiết
            if st["alive"]:
                stat_txt = f"Bom: {agent.bombs_placed}/{agent.ammo} | Tầm nổ: {agent.blast_range} | Phá: {st['walls']} | Kills: {st['kills']}"
                stat_surf = self.font_small.render(stat_txt, True, COLOR_TEXT_MUTED)
            else:
                stat_surf = self.font_small.render("ĐÃ BỊ TIÊU DIỆT", True, (255, 85, 85))
            self.screen.blit(stat_surf, (av_x + 20, py + 30))

            py += card_h + 8

        # 3. Nhật ký sự kiện (Event Log)
        py += 4
        log_title = self.font_small.render("NHẬT KÝ CHIẾN ĐẤU (EVENT LOG):", True, COLOR_TEXT_MUTED)
        self.screen.blit(log_title, (px, py))
        py += 18

        log_box = pygame.Rect(px, py, dw - 36, 110)
        pygame.draw.rect(self.screen, (24, 27, 38), log_box, border_radius=6)
        pygame.draw.rect(self.screen, (40, 45, 65), log_box, width=1, border_radius=6)

        log_lines = self.controller.event_log[:5]
        for l_idx, line in enumerate(log_lines):
            l_surf = self.font_small.render(line, True, COLOR_TEXT_MAIN if l_idx == 0 else COLOR_TEXT_MUTED)
            self.screen.blit(l_surf, (px + 10, py + 8 + l_idx * 19))
        py += 122

        # 4. Hướng dẫn phím nóng (Controls Toolbar)
        if self.controller.agent_types[0] == "HUMAN":
            ctrl_txt = "[W-A-S-D / Mũi tên]: Di chuyển | [SPACE / ENTER]: Đặt bom"
        else:
            ctrl_txt = "[SPACE]: Dừng/Chạy | [N]: Bước tiếp | [1-4]: Đổi tốc độ"
        ctrl_txt2 = "[R]: Ván mới | [S]: Sudden Death | [M]: Menu chính"

        c1 = self.font_small.render(ctrl_txt, True, (255, 215, 64) if self.controller.agent_types[0] == "HUMAN" else COLOR_TEXT_MUTED)
        c2 = self.font_small.render(ctrl_txt2, True, COLOR_TEXT_MUTED)
        self.screen.blit(c1, (px, py))
        self.screen.blit(c2, (px, py + 18))


# ───────────────────────────── KHỞI CHẠY ─────────────────────────────

if __name__ == "__main__":
    app = PygameApp()
    app.run()
