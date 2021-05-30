class custom_console:
    @staticmethod
    def build_row(line, max_width=30):
        result_line = ''
        for i in line:
            spaces = max_width - len(str(i))
            result_line += str(i)
            result_line += ' ' * spaces
        return result_line
