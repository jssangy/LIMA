from map import map6


def convert_all_str_to_6(map_data):
    """리스트 내 모든 문자열을 6으로 변환"""
    return [
        [6 if isinstance(cell, str) else cell for cell in row]
        for row in map_data
    ]

def print_map_list_as_txt_format(map_data):
    """maps.txt 포맷으로 출력"""
    print("map = [")
    for row in map_data:
        line = "    [ " + "  , ".join(f"{cell:2d}" for cell in row) + "  ],"
        print(line)
    print("]")

print_map_list_as_txt_format(convert_all_str_to_6(map6))