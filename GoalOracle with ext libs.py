import pygame
from pygame import MOUSEBUTTONDOWN, Surface

pygame.init()

screen_width = 800
screen_height = 600

print("Thanks for trying out our program, hope you enjoyed! : )")
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("GoalOracle:- Home")
font= pygame.font.Font(None, 30)
text= font.render("PREDICT", True, 'Black')
exitext= font.render("EXIT", True, "Black")
background_color = (0, 128, 255)
button= pygame.Rect(100,150,150,50)
exitbutton= pygame.Rect(500,150,150,50)

pygame.display.flip()


def game():
    import pygame
    import numpy as np
    from scipy.stats import poisson
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as agg

    BLACK = (10, 10, 10)
    GOLD = (212, 175, 55)
    BEIGE = (245, 245, 220)
    CREAM = (255, 253, 208)
    WHITE = (255, 255, 255)
    DARK_GREY = (40, 40, 40)
    GRAY = (70, 70, 70)

    pygame.init()
    window_size = (1130, 770)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("GoalOracle ⚽")
    font = pygame.font.SysFont("Arial", 22)
    big_font = pygame.font.SysFont("Arial", 28, bold=True)
    panel_font = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()
    bg = pygame.image.load('Bagoroucknd.png')
    bg = pygame.transform.smoothscale(bg, (1130,770))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    # Logo load
        try:
            logo = pygame.image.load("Goal Oracle.png").convert_alpha()
            logo = pygame.transform.smoothscale(logo, (130, 130))
        except:
            logo = None

        stats_keys = [
            ("goals_scored", "Goals Scored"),
            ("goals_conceded", "Goals Conceded"),
            ("shots_on_target", "Shots on Target"),
            ("chances_created", "Chances Created"),
            ("possession", "Possession"),
            ("pass_completion", "Pass Completion")
        ]

        class InputBox:
            def __init__(self, x, y, w=260, h=30):
                self.rect = pygame.Rect(x, y, w, h)
                self.color_active = GOLD
                self.color_inactive = DARK_GREY
                self.color = self.color_inactive
                self.text = ''
                self.txt_surface = font.render(self.text, True, WHITE)
                self.active = False

            def handle_event(self, event):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.active = self.rect.collidepoint(event.pos)
                    self.color = self.color_active if self.active else self.color_inactive
                if event.type == pygame.KEYDOWN and self.active:
                    if event.key == pygame.K_RETURN:
                        self.active = False
                        self.color = self.color_inactive
                    elif event.key == pygame.K_BACKSPACE:
                        self.text = self.text[:-1]
                    else:
                        if event.unicode.isdigit() or (event.unicode == '.' and '.' not in self.text):
                            self.text += event.unicode
                    self.txt_surface = font.render(self.text, True, WHITE)

            def draw(self, screen):
                pygame.draw.rect(screen, self.color, self.rect, border_radius=5)
                text_y = self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2
                screen.blit(self.txt_surface, (self.rect.x + 8, text_y))
                pygame.draw.rect(screen, GRAY, self.rect, 2, border_radius=5)

            def get_value(self):
                try:
                    return float(self.text)
                except:
                    return None

            def clear(self):
                self.text = ''
                self.txt_surface = font.render(self.text, True, WHITE)

        class Button:
            def __init__(self, x, y, w, h, text):
                self.rect = pygame.Rect(x, y, w, h)
                self.text = text
            def draw(self, surface):
                txt_surf = big_font.render(self.text, True, WHITE)
                pygame.draw.rect(surface, GRAY, self.rect, border_radius=5)
                surface.blit(txt_surf, ((self.rect.centerx - 5) - txt_surf.get_width() // 2,
                                        self.rect.centery - txt_surf.get_height() // 2))
            def is_clicked(self, event):
                return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

        def calculate_score_probabilities(lambda_a, lambda_b, max_goals=5):
            matrix = np.zeros((max_goals + 1, max_goals + 1))
            for i in range(max_goals+1):
                for j in range(max_goals+1):
                    matrix[i][j] = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
            return matrix

        def calculate_outcome_probabilities(prob_matrix):
            win_a = np.tril(prob_matrix, -1).sum()
            draw = np.trace(prob_matrix)
            win_b = np.triu(prob_matrix, 1).sum()
            return win_a, draw, win_b

        def draw_background(surface):
            screen.blit(bg, (0,0))

        input_boxes = {
            "Team A": {},
            "Team B": {}
        }

        # Layout coordinates as per screenshot
        team_a_x = 180
        team_b_x = 660
        inputs_y = [235, 285, 335, 385, 435, 485]

        for idx, (key, label) in enumerate(stats_keys):
            input_boxes["Team A"][key] = InputBox(team_a_x, inputs_y[idx], 260, 36)
            input_boxes["Team B"][key] = InputBox(team_b_x, inputs_y[idx], 260, 36)

        labels_y = [y - 22 for y in inputs_y]
        button_y = 570
        predict_btn = Button(window_size[0]//2 - 140, button_y, 180, 50, "Predict")
        reset_btn = Button(window_size[0]//2 + 25, button_y, 180, 50, "Reset")

        panel_rect = pygame.Rect(0, 645, window_size[0], 95)
        result_lines = ["Most Probable Score: -", "Team A Win: - | Draw: - | Team B Win: -"]

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                for team in input_boxes:
                    for key in input_boxes[team]:
                        input_boxes[team][key].handle_event(event)
                if predict_btn.is_clicked(event):
                    try:
                        team_a_stats = {k: input_boxes["Team A"][k].get_value() for k, _ in stats_keys}
                        team_b_stats = {k: input_boxes["Team B"][k].get_value() for k, _ in stats_keys}
                        if None in team_a_stats.values() or None in team_b_stats.values():
                            raise ValueError("Invalid input detected")
                        lambda_a = team_a_stats["goals_scored"]
                        lambda_b = team_b_stats["goals_scored"]
                        prob_matrix = calculate_score_probabilities(lambda_a, lambda_b)
                        win_a, draw, win_b = calculate_outcome_probabilities(prob_matrix)
                        max_prob = np.max(prob_matrix)
                        max_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
                        result_lines = [
                            f"Most Probable Score: {max_idx[0]} - {max_idx[1]} ({max_prob:.2%})",
                            f"Team A Win: {win_a:.2%} | Draw: {draw:.2%} | Team B Win: {win_b:.2%}",
                        ]
                    except Exception:
                        result_lines = ["Invalid input detected. Please correct entries.", ""]
                if reset_btn.is_clicked(event):
                    for team in input_boxes:
                        for key in input_boxes[team]:
                            input_boxes[team][key].clear()
                    result_lines = ["Most Probable Score: -", "Team A Win: - | Draw: - | Team B Win: -"]

            draw_background(screen)
            if logo:
                screen.blit(logo, (window_size[0]//2 - logo.get_width()//2, 38))
                pygame.draw.rect(screen, (50,40,20), (window_size[0]//2 - logo.get_width()//2, 38, 130, 130),8, border_radius= 200)
            team_a_lbl = big_font.render("Team A", True, WHITE)
            team_b_lbl = big_font.render("Team B", True, WHITE)
            screen.blit(team_a_lbl, (team_a_x + 60, 185))
            screen.blit(team_b_lbl, (team_b_x + 60, 185))
            for i, (key, label) in enumerate(stats_keys):
                label_surf = font.render(label, True, WHITE)
                screen.blit(label_surf, (team_a_x + 5, labels_y[i]))
                input_boxes["Team A"][key].draw(screen)
                screen.blit(label_surf, (team_b_x + 5, labels_y[i]))
                input_boxes["Team B"][key].draw(screen)
            predict_btn.draw(screen)
            reset_btn.draw(screen)
            pygame.draw.rect(screen, DARK_GREY, panel_rect, border_radius=20)
            pygame.draw.rect(screen, CREAM, panel_rect, 5, border_radius=20)
            for i, line in enumerate(result_lines):
                txt_surf = panel_font.render(line, True, WHITE)
                screen.blit(txt_surf, (51, 663 + i * 28))
            pygame.draw.rect(screen, CREAM, (window_size[0] // 2 + 30, button_y, 175, 50), 5)
            pygame.draw.rect(screen, CREAM, (window_size[0]//2 - 145, button_y, 175, 50), 5)
            pygame.display.flip()
            clock.tick(60)
        return


run = True
while run:
   for event in pygame.event.get():
       if event.type == pygame.QUIT:
           pygame.quit()
   if event.type == MOUSEBUTTONDOWN:
       if exitbutton.collidepoint(event.pos):
           pygame.quit()
   if event.type == MOUSEBUTTONDOWN:
       if button.collidepoint(event.pos):
           game()

   bg2 = pygame.image.load('Backgorund.png')
   screen.blit(bg2, (0,0))
   pygame.draw.rect(screen, 'beige', button, border_radius=10)
   screen.blit(text, (button.x + 30, button.y + 15))
   pygame.draw.rect(screen, 'beige', exitbutton, border_radius=10)
   screen.blit(exitext, (exitbutton.x + 50, exitbutton.y + 15))
   pygame.display.flip()

