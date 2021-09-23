import numpy as np
import cv2
import multiprocessing as mp

from Agent.PieceClassifier import PieceClassifier
from Controller.MoveDescription import MoveDescription
from Observation.Gamestate import GamestateStruct
from FrameGrabber.Frame import Frame

from Helpers.Draw import Draw
from Helpers.InputPool import InputPool


class GamestateRenderer:
    def __init__(self):
        self.target_move:MoveDescription = None
        self.state = GamestateStruct()
        self.fps = 0

        self.inputpool:InputPool = None
        self.queue = mp.JoinableQueue()
        self._renderer_process = mp.Process(target=self.renderer_loop, args=(InputPool(),))
        self._renderer_process.start()

    def stop(self):
        self._renderer_process.kill()

    def process(self, state, target_move, fps):
        self.queue.put((state, target_move, fps))
        
    def renderer_loop(self, inputpool):
        self.inputpool = inputpool

        while True:
            if self.queue.empty():
                self._render()
            
            else:
                inputs = self.queue.get()
                self.state = inputs[0]
                self.target_move = inputs[1]
                self.fps = inputs[2]
                self._render()

                self.queue.task_done()

    def _render(self):
        if self.state is None:
            self.state = GamestateStruct()

        # try:
        frame = self.render_state(self.state, self.target_move)
        frame = self.draw_fps(frame, self.fps)
        frame = self.draw_active_piece_lifetime(frame, self.state.active_piece_lifetime)
        frame = self.draw_game_lifetime(frame, self.state.game_lifetime)
        frame = self.draw_target_move_pos(frame, self.target_move)

        cv2.imshow("gamestate", frame)
        if (last_key := cv2.waitKey(1)) != -1:
            self.inputpool.add_key(last_key)
        # cv2.waitKey(1)
        # except Exception:
        #     print("-"*60)
        #     traceback.print_exc(file=sys.stdout)
        #     print("-"*60)  

    def draw_fps(self, frame, fps):
        draw = Draw.begin(frame)
        msg = f"FPS: {fps:.0f}"
        Draw.text(draw, msg, (25, 25))
        return Draw.end(draw)

    def draw_active_piece_lifetime(self, frame, active_piece_lifetime):
        draw = Draw.begin(frame)
        msg = f"active lifetime: \n{active_piece_lifetime:.2f}"
        Draw.text(draw, msg, (25, 50))
        return Draw.end(draw)

    def draw_game_lifetime(self, frame, game_lifetime):
        draw = Draw.begin(frame)
        msg = f"game_lifetime: \n{game_lifetime:.2f}"
        Draw.text(draw, msg, (25, 90))
        return Draw.end(draw)

    def draw_target_move_pos(self, frame, target_move:MoveDescription):
        if target_move and target_move.destination:
            draw = Draw.begin(frame)
            pos = target_move.destination.position
            msg = f"target_pos: \n{pos}"
            Draw.text(draw, msg, (25, 130))
            return Draw.end(draw)
        return frame

    def place_piece(self, blocks_in, piece, color_value=2):
        blocks = blocks_in.copy()
        h, w = piece.blocks.shape
        x, y = piece.position

        for row_i, layer in enumerate(piece.blocks):
            # print(row_i, layer)
            for col_i, col_value in enumerate(layer):
                if col_value == 0:
                    continue
                
                py, px = (row_i+(y-h+1), col_i+x)
                if px >= 10 or py >= 23:
                    return blocks_in

                # print((row_i+y, col_i+x), (px, py), col_value)
                if py >= 0 and px >= 0:
                    blocks[py, px] = color_value

        return blocks

    def render_state(self, state:GamestateStruct, target_move:MoveDescription):
        held_im = np.zeros((4, 6, 3), dtype=np.int8)
        if state.held_piece:
            coords = list(np.where(state.held_piece.blocks > 0))
            coords[0] += 1
            coords[1] += (6 - state.held_piece.width) // 2
            held_im[tuple(coords)] = (150, 148, 140) if state.is_held_disabled else (128, 200, 200)
        held_im = Frame.add_padding(held_im, right=1)

        gameboard_im = state.dead_blocks

        if state.active_piece is not None:
            ## Overlay the active piece onto the gameboard
            gameboard_im = self.place_piece(gameboard_im, state.active_piece, 4)

        if target_move is not None:
            target = target_move.destinations[-1]
            if target is not None:
                gameboard_im = self.place_piece(gameboard_im, target, 2)

        next_queue_im = np.zeros((22, 6, 3), dtype=np.int8)

        coords_collection = []
        if state.next_queue is not None:
            for idx, each_piece_name in enumerate(state.next_queue):
                each_piece = PieceClassifier().get_piece(each_piece_name)
                _, w = each_piece.blocks.shape

                coords = list(np.where(each_piece.blocks > 0))
                coords[0] += (2 * idx) + (idx * 2) + 1
                coords[1] += (6 - w) // 2
                coords_collection.append(np.array(coords))

        if len(coords_collection) > 0:
            coords_max = np.max(coords_collection[-1], axis=0)
            if len(coords_max) > 0:
                height = max(coords_max) + 2
                next_queue_im = np.zeros((height+1, 6, 3), dtype=np.uint8)

                for coords in coords_collection:
                    next_queue_im[tuple(coords)] = (128, 200, 200)

        gameboard_im = Frame.change_nchannels(gameboard_im, 3)
        gameboard_im[gameboard_im[:,:,0] == 1] = (255, 255, 255)
        gameboard_im[gameboard_im[:,:,0] == 2] = (141, 46, 230)
        gameboard_im[gameboard_im[:,:,0] == 3] = (46, 224, 230)
        gameboard_im[gameboard_im[:,:,0] == 4] = (40, 209, 102)
        gameboard_im[gameboard_im[:,:,0] == 5] = (60, 60, 255)
        gameboard_im = Frame.add_padding(gameboard_im, right=2)

        gamestate_im = Frame.hstack((held_im, gameboard_im, next_queue_im))
        gamestate_im = Frame.resize(gamestate_im, 400)
        gamestate_im = Frame.add_padding(gamestate_im, 25, 25, 25, 25)
        return gamestate_im
