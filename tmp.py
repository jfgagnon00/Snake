    def _setFoodInGrid(self, show):
        # validation
        if not (0 <= self._food.x and self._food.x < self._occupancyGridWidth):
            raise ResetException("Positon x food invalide", None, self._food.copy())

        if not (0 <= self._food.y and self._food.y < self._occupancyGridHeight):
            raise ResetException("Positon y food invalide", None, self._food.copy())

        expectedOccupancy = GridOccupancy.EMPTY if show else GridOccupancy.FOOD
        if self._occupancyGrid[self._food.y, self._food.x] != expectedOccupancy:
            raise ResetException("Occupancy food invalide",
                                 self._occupancyGrid.copy(),
                                 self._food.copy())

        # placer dans la grille
        self._occupancyGrid[self._food.y, self._food.x] = GridOccupancy.FOOD if show else GridOccupancy.EMPTY





    def _setSnakeInGrid(self, show):
        # sous optimal, a changer
        value = GridOccupancy.SNAKE_BODY if show else GridOccupancy.EMPTY

        for i, p in enumerate(state.snake.bodyParts):
            if show and self._occupancyGrid[p.y, p.x] != GridOccupancy.EMPTY:
                raise ResetException("Placement snake bodyParts invalide",
                                     self._occupancyGrid.copy(),
                                     p.copy())

            self._occupancyGrid[p.y, p.x] = value

        if show:
            tail = state.snake.tail

            if self._occupancyGrid[tail.y, tail.x] != GridOccupancy.SNAKE_BODY:
                raise ResetException("Placement snake tail invalide",
                                     self._occupancyGrid.copy(),
                                     tail.copy())

            self._occupancyGrid[tail.y, tail.x] = GridOccupancy.SNAKE_TAIL

            head = state.snake.head

            if self._occupancyGrid[head.y, head.x] != GridOccupancy.SNAKE_BODY:
                raise ResetException("Placement snake head invalide",
                                     self._occupancyGrid.copy(),
                                     head.copy())

            self._occupancyGrid[head.y, head.x] = GridOccupancy.SNAKE_HEAD
            self._occupancyGridCount[head.y, head.x] += 1

