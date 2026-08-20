import os
import builtins

CONSOLE_WIDTH, CONSOLE_HEIGHT = os.get_terminal_size()

class Graph:
    def __init__(
        self,
        x_axis_label: builtins.str = 'Iterations',
        y_axis_label: builtins.str = 'Loss',
        graph_height: builtins.int = 20,
        graph_width:  builtins.int = 80
    ) -> None:
        self.x_axis_label = x_axis_label.upper()
        self.y_axis_label = y_axis_label.upper()
        self.graph_height = graph_height
        self.graph_width  = graph_width
        
        self.data         = []
        self.x_max_value  = 100.0
        self.y_max_value  = 5.0
        self.num_values_x = 5
        self.num_values_y = 5
        
        self.x_buffer      = 6
        self.y_buffer      = 3
        self.graph_width  += self.x_buffer
        self.graph_height += self.y_buffer
        
        self.charset: list[builtins.str] = ['█', '▄']
        self.canvas:  list[builtins.str] = []
        self._init_canvas()
    
    def _init_canvas(self) -> None:
        for _ in range(self.graph_height):
            row = [' ' for _ in range(self.graph_width)]
            self.canvas.append(row)
    
    def _build_base(self) -> None:
        for r_idx, row in enumerate(self.canvas):
            H = self.graph_height-self.y_buffer

            for c_idx in range(len(row)):
                
                # Add vertical line
                if c_idx == self.x_buffer and r_idx < H:
                    self.canvas[r_idx][c_idx] = self.charset[0]
                    
                # Add horizonal line
                elif r_idx == H - 1 and c_idx >= self.x_buffer:
                    self.canvas[r_idx][c_idx] = self.charset[1]
            
            if r_idx <= H:
                if r_idx % 2 == 0:
                    value_y = str((1.0 - (r_idx / H)) * self.y_max_value).zfill(3)
                    self.canvas[r_idx][2] = value_y[0]    
                    self.canvas[r_idx][3] = value_y[1]
                    self.canvas[r_idx][4] = value_y[2]
            
        y_label_len = len(self.y_axis_label)
        y_label_start = (
            self.graph_height 
          - y_label_len
          - self.y_buffer
        ) // 2
        for i in range(y_label_len):
            self.canvas[y_label_start+i][0] = self.y_axis_label[i]
            
        x_label_len = len(self.x_axis_label)
        x_label_start = (
            self.graph_width 
          - x_label_len
          + self.x_buffer
        ) // 2
        for i in range(x_label_len):
            self.canvas[self.graph_height-1][i+x_label_start] = self.x_axis_label[i]
    
    
    def _build_canvas(self) -> None:
        self._build_base()
        # for r_idx, row in enumerate(self.canvas):
        #     for c_idx in range(len(row)):
        #         self.canvas[r_idx][c_idx] = '■'
        
        
    def draw(self) -> None:
        self._build_canvas()
        for row in self.canvas:
            line = ''.join(row)
            print(line)
                
                
                
if __name__ == '__main__':
    graph = Graph()
    graph.draw()





