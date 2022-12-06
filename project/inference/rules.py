'''
rule 설계 파이썬 파일
작성담당자: 박희용대리님, 손명원대리님
해당 파이썬 구조는 얼마든지 구조변경 가능합니다!
'''
import numpy as np

class Rules:
    def __init__(self):
        self.rule_structure = {
            "address": {
                "어깨선각도  [어깨 각도(°)]": 0.0,
                "양손위치  [양발간격 기준 양손위치 (％)]": 0.0,
                "스탠스  [어깨대비 양발간격 비율(％)]": 0.0,
                "상체기울임정도  [상체 길이 대비 하체 길이 비율(％)]": 0.0
            },
            "backswing": {
                "왼쪽어깨회전  [무릎간격 기준 왼쪽 어깨위치(％)]": 0.0,
                "왼팔펴짐각도  [왼팔 각도(°)]": 0.0
            },
            "top": {
                "하체고정  [오른무릎-오른골반-왼무릎 각도(°)]": 0.0,
                "오른쪽다리펴짐각도  [오른쪽 골반-무릎-발 각도(°)]": 0.0,
                "스웨이체크  [무릎간격 기준 어깨 중점 위치(％)]": 0.0
            },
            "impact": {
                "행잉백  [왼쪽 어깨-무릎선과 Y축 각도(°)]": 0.0,
                "오른쪽K라인각도  [오른쪽 어깨-골반-발 각도(°)]": 0.0,
                "왼팔펴짐각도  [왼팔 각도(°)]": 0.0,
                "오른팔펴짐각도  [오른팔 각도(°)]": 0.0
            },
            "follow": {
                "왼쪽라인  [왼쪽 골반-무릎선과 Y축 각도(°)]": 0.0,
                "치킨윙  [왼팔 각도(°)]": 0.0,
                "체중전진이동  [발간격 기준 골반 중점 위치(％)]": 0.0,
                "오른쪽다리펴짐각도  [오른쪽 골반-무릎-발 각도(°)]": 0.0
            }
        }
    
    def get_grade(self, x: float, label: dict):
        p_1sigma = label['mean'] + label['std']
        p_2sigma = label['mean'] + label['std'] * 2
        m_1sigma = label['mean'] - label['std']
        m_2sigma = label['mean'] - label['std'] * 2
        if (m_1sigma < x) and (x < p_1sigma):
            return '우수'
        elif (m_2sigma < x and x < m_1sigma) or (p_1sigma < x and x < p_2sigma):
            return '보통'
        else:
            return '교정필요'
    
    def handle(self, action: str, value: dict, coordis: dict):
        if action == 'address':
            value[action]['어깨선각도  [어깨 각도(°)]'] = self.tan_slope(coordis['left_shoulder'], coordis['right_shoulder'])
            res = self.center_calc(coordis['left_wrist'], coordis['right_wrist'])
            value[action]['양손위치  [양발간격 기준 양손위치 (％)]'] = self.ratio_interpolation(coordis['left_ankle'],coordis['right_ankle'], res)
            value[action]['스탠스  [어깨대비 양발간격 비율(％)]'] = self.ratio(coordis['left_shoulder'], coordis['right_shoulder'], coordis['left_heel'], coordis['right_heel'])
            value[action]['상체기울임정도  [상체 길이 대비 하체 길이 비율(％)]'] = self.ratio(coordis['right_heel'], coordis['right_hip'], coordis['right_hip'], coordis['right_shoulder'])
        elif action == 'backswing':
            value[action]['왼쪽어깨회전  [무릎간격 기준 왼쪽 어깨위치(％)]'] = self.ratio_interpolation(coordis['right_ankle'], coordis['left_ankle'], coordis['left_shoulder'])
            value[action]['왼팔펴짐각도  [왼팔 각도(°)]'] = self.angle(coordis['left_wrist'], coordis['left_elbow'], coordis['left_shoulder'])
        elif action == 'top':
            value[action]['하체고정  [오른무릎-오른골반-왼무릎 각도(°)]'] = self.angle(coordis['right_knee'], coordis['right_hip'], coordis['left_knee'])
            value[action]['오른쪽다리펴짐각도  [오른쪽 골반-무릎-발 각도(°)]'] = self.angle(coordis['right_hip'], coordis['right_knee'], coordis['right_ankle'])
            res = self.center_calc(coordis['left_shoulder'], coordis['right_shoulder'])    
            value[action]['스웨이체크  [무릎간격 기준 어깨 중점 위치(％)]'] = self.ratio_interpolation(coordis['right_ankle'], coordis['left_ankle'], res)
        elif action == 'impact':
            value[action]['행잉백  [왼쪽 어깨-무릎선과 Y축 각도(°)]'] = self.angle_2(coordis['left_shoulder'], coordis['left_ankle'])
            value[action]['오른쪽K라인각도  [오른쪽 어깨-골반-발 각도(°)]'] = self.angle(coordis['right_shoulder'], coordis['right_hip'], coordis['right_ankle'])
            value[action]['왼팔펴짐각도  [왼팔 각도(°)]'] = self.angle(coordis['left_wrist'], coordis['left_elbow'], coordis['left_shoulder'])
            value[action]['오른팔펴짐각도  [오른팔 각도(°)]'] = self.angle(coordis['right_wrist'], coordis['right_elbow'], coordis['right_shoulder'])
        elif action == 'follow':
            value[action]['왼쪽라인  [왼쪽 골반-무릎선과 Y축 각도(°)]'] = self.angle_2(coordis['left_hip'], coordis['left_ankle'])
            value[action]['치킨윙  [왼팔 각도(°)]'] = self.angle(coordis['left_wrist'], coordis['left_elbow'], coordis['left_shoulder'])     
            value[action]['체중전진이동  [발간격 기준 골반 중점 위치(％)]'] = self.ratio_interpolation(coordis['right_ankle'], coordis['left_ankle'],cal_point=np.array([0,0,0]) )
            value[action]['오른쪽다리펴짐각도  [오른쪽 골반-무릎-발 각도(°)]'] = self.angle(coordis['right_hip'], coordis['right_knee'], coordis['right_ankle'])
        return value
            
    def tan_slope(self, a: np.ndarray, b: np.ndarray):
        '''
        두 Points 가 주어졌을 때 두 point간 vector를 구하고 vector와 x축간 기울어진 정도를 구하는 함수
        **args**
        - a : 첫 번째 point의 vector(np.ndarray) 
        - b : 두 번째 point의 vector(np.ndarray) 
        **return**
        - vector와 x축간 기울어진 정도(float, 0~180)
        '''
        x = b[0] - a[0]
        y = abs(b[1] - a[1])
        slope = abs(180 / np.pi * np.arctan(y / x))
        return round(slope, 2)

    def angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        '''
        세 Points 가 주어졌을 때 두번째 point에서 형성되는 각도를 구하는 함수
        **args**
        - a : 첫 번째 point의 vector(np.ndarray) 
        - b : 두 번째 point의 vector(np.ndarray)
        - c : 세 번째 point의 vector(np.ndarray)
        **return**
        - 두번째 point에서 형성되는 각도(float)
        '''
        v1 = c - b
        v2 = a - b
        v1 = v1[:2]
        v2 = v2[:2]
        angle_cos = (v1 @ v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))
        angle = 180 / np.pi * np.arccos(angle_cos)
        return round(angle, 2)

    def angle_2(self, a: np.ndarray, b: np.ndarray):
        '''
        두 Points 가 주어졌을 때 y축방향의 가상의 line을 긋고 가상의 line에서 가상의 point를 정하여 두번째 point에서 형성되는 각도를 구하는 함수
        가상의 point(new point)는 첫번째 점의 x좌표값과 두번째 점의 y좌표값으로 구한다
        **args**
        - a : 첫 번째 point의 vector(np.ndarray) 
        - b : 두 번째 point의 vector(np.ndarray)
        **return**
        - 두번째 point에서 형성되는 각도(float)
        '''
        new_point = [b[0],a[1],0]
        v1 = a-b
        v2 = new_point-b
        v1 = v1[:2]
        v2 = v2[:2]
        angle_cos = (v1@ v2)/(np.linalg.norm(v1,2)*np.linalg.norm(v2,2))
        angle = 180/np.pi*np.arccos(angle_cos)
        return round(angle,2)

    def ratio(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, dim=3):
        '''
        4개 vectors가 주어졌을 때 첫번째 두번째 vectors pair와 세번째 네번째 vector pair간 비율을 구하는 함수
        **args**
        - (a,b) : 첫번째 vector 연산을 위한 2개 vectors(np.ndarray) 
        - (c,d) : 두번째 vector 연산을 위한 2개 vectors(np.ndarray)
        - dim : 계산하고자 하는 차원수
        **return**
        - 첫번째 vector 대비 두번째 vector 비율(float)
        '''
        r1_vector = c[:dim] - d[:dim]
        r1_norm = np.linalg.norm(r1_vector,2)
        r2_vector = a[:dim] - b[:dim]
        r2_norm = np.linalg.norm(r2_vector,2)
        return round(abs(r2_norm/r1_norm*100),2)

    def ratio_interpolation(self, a: np.ndarray,b: np.ndarray, cal_point: np.ndarray):
        '''
        3개 vectors가 주어졌을 때 첫번째 두번째 vectors pair와 cal_point와 첫번째 vector간 비율을 구하는 함수
        **args**
        - (a,b) : vector 연산을 위한 2개 vectors(np.ndarray) 
        - cal_point : interpolation point
        - dim : 계산하고자 하는 차원수
        **return**
        - 첫번째 vector 대비 두번째 vector 비율(float)
        '''
        r1_vector = a[0] - b[0]
        r2_vector = a[0] - cal_point[0]
        return round(abs(r2_vector/r1_vector*100),2)

    def center_calc(self, start:np.ndarray, end:np.ndarray):
        res = start + ((end-start)*0.5)
        return res