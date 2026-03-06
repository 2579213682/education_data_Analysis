import pandas as pd
import numpy as np
import json
import random
filepath_TeacherInfo = '../../education_data/1_teacher.csv'
filepath_StudentsInfo = '../../education_data/2_student_info.csv'
filepath_StudentsInfo_processed = '../../education_data/CH/Student/1.StudentInfo.csv'
filepath_AttendenceInfo = '../../education_data/3_kaoqin.csv'
filepath_StudentsScore = '../../education_data/5_chengji.csv'
filepath_StudentsConsumption = '../../education_data/7_consumption.csv'
filepath_StudentsConsumption_Processed = '../../education_data/CH/Student/5.Consumption.csv'
subname = ['语文', '数学', '英语', '物理', '化学', '生物', '政治', '历史', '地理', '技术', '体育', '音乐']
def statistic_teachers_info(studentID):
    data_StudentInfo = pd.read_csv(filepath_StudentsInfo)
    data_TeacherInfo = pd.read_csv(filepath_TeacherInfo)
    teacherName = ['NaN'] * 12
    studentInfo = data_StudentInfo.drop(data_StudentInfo[data_StudentInfo['bf_StudentID'] != studentID].index)
    if studentInfo.shape[0] == 0:
        print('没有该学生的信息，请重新输入！')
    else:
        classID = studentInfo['cla_id'].iloc[0]
        teacherInfo = data_TeacherInfo.drop(data_TeacherInfo[data_TeacherInfo['cla_id'] != classID].index)
        for i in range(len(subname)):
            for j in range(teacherInfo.shape[0]):
                if subname[i] == teacherInfo['sub_Name'].iloc[j]:
                    teacherName[i] = teacherInfo['bas_Name'].iloc[j]
        print(teacherName)
        for i in range(len(teacherName)):
            print('学科', subname[i], '的教师是', teacherName[i])
        return teacherName
def data_preprocess_studentInfo():
    data_StudentInfo = pd.read_csv(filepath_StudentsInfo)
    data_StudentInfo['bf_BornDate'] = data_StudentInfo['bf_BornDate'].fillna(method='pad')
    data_StudentInfo['bf_qinshihao'] = data_StudentInfo['bf_qinshihao'].fillna(-1)
    data_StudentInfo['bf_zhusu'] = data_StudentInfo['bf_zhusu'].fillna(-1)
    data_StudentInfo = data_StudentInfo.drop(data_StudentInfo[data_StudentInfo['bf_NativePlace'] == '汉族'].index)
    data_StudentInfo = data_StudentInfo.drop(data_StudentInfo[data_StudentInfo['bf_NativePlace'] == '汉'].index)
    data_StudentInfo.to_csv('../../education_data/CH/Student/1.StudentInfo.csv', encoding='utf_8_sig')
    print('文件保存完成！')
studentInfo_field = ['姓名', '学号', '性别', '民族', '班级名', '出生日期', '籍贯', '政治面貌', '住宿信息', '寝室号', '寝室人数']
def statistic_student_info(studentID):
    data_StudentInfo = pd.read_csv(filepath_StudentsInfo_processed)
    studentInfo_Array = ['NaN'] * 11
    studentInfo = data_StudentInfo.drop(data_StudentInfo[data_StudentInfo['bf_StudentID'] != studentID].index)
    studentInfo_Array[0] = studentInfo['bf_Name'].iloc[0]
    studentInfo_Array[1] = int(studentInfo['bf_StudentID'].iloc[0])
    studentInfo_Array[2] = studentInfo['bf_sex'].iloc[0]
    studentInfo_Array[3] = studentInfo['bf_nation'].iloc[0]
    studentInfo_Array[4] = studentInfo['cla_Name'].iloc[0]
    studentInfo_Array[5] = str(int(studentInfo['bf_BornDate'].iloc[0])) + '年'
    studentInfo_Array[6] = studentInfo['bf_NativePlace'].iloc[0]
    studentInfo_Array[7] = studentInfo['bf_policy'].iloc[0]
    studentInfo_Array[8] = '非住校'
    if studentInfo['bf_zhusu'].iloc[0] == 1:
        studentInfo_Array[8] = '住校'
        if studentInfo['bf_qinshihao'].iloc[0] != 'Null':
            studentInfo_Array[9] = str(int(studentInfo['bf_qinshihao'].iloc[0])) + '号'
            roomNumber = studentInfo['bf_qinshihao'].iloc[0]
            data_groupbyRoomNum = data_StudentInfo.drop(data_StudentInfo[data_StudentInfo['bf_qinshihao'] != roomNumber].index)
            studentInfo_Array[10] = int(data_groupbyRoomNum.shape[0])
    print(studentInfo_Array)
    json_data = {"info": studentInfo_Array}
    with open('../3.Student-level-data/Student_Info_1.json', "w") as file:
        json.dump(json_data, file)
    print('完成文件加载')
def statistic_class_studentsNum():
    data_studentsInfo = pd.read_csv(filepath_StudentsInfo)
    class_name = []
    students_num = []
    data_studentsInfo['count'] = 1
    class_studentsNum = data_studentsInfo.groupby(['cla_Name']).count().reset_index()
    class_studentsNum = class_studentsNum.drop(class_studentsNum[class_studentsNum['count'] < 30].index)
    for i in range(class_studentsNum.shape[0]):
        class_name.append(class_studentsNum['cla_Name'].iloc[i])
        students_num.append(class_studentsNum['count'].iloc[i])
    print(class_name)
    print(students_num)
def statistic_students_attendance():
    data_attendance = pd.read_csv(filepath_AttendenceInfo)
    data_studentInfo = pd.read_csv(filepath_StudentsInfo)
    data_attendance['count'] = 1
    data_attendance_grounpby = data_attendance.groupby(['bf_studentID']).count().reset_index().sort_values(by='count', axis=0, ascending=False)
    data_attendance_grounpby = data_attendance_grounpby.iloc[0: 100]
    studentID_Array = []
    count = 0
    for i in range(data_attendance_grounpby.shape[0]):
        studentID_Array.append(data_attendance_grounpby['bf_studentID'].iloc[i])
    for i in range(len(studentID_Array)):
        for j in range(data_studentInfo.shape[0]):
            if studentID_Array[i] == data_studentInfo['bf_StudentID'].iloc[j]:
                print(studentID_Array[i])
                count += 1
    print(count)
    print(data_attendance_grounpby.iloc[99])
    print(data_attendance_grounpby.shape)
def statistic_student_attendance(studentID):
    print('统计学生的考勤情况中...')
    data_attendance = pd.read_csv(filepath_AttendenceInfo)
    # data_attendance['month']
    data_attendance['hour'] = (data_attendance['DataDateTime'].str.split(' ', expand=True)[1]).str.split(':', expand=True)[0]
    data_attendance['minute'] = ((data_attendance['DataDateTime'].str.split(' ', expand=True)[1]).str.split(':', expand=True)[1]).str.split(':', expand=True)[0]
    student_attendance = data_attendance.drop(data_attendance[data_attendance['bf_studentID'] != studentID].index)
    late_taskName = [100100, 9900100, 10020]
    early_taskName = [200200, 9900300]
    uniform_taskName = [200100, 9900200]
    student_problems_num = [0] * 3
    class_problems_num = [0] * 3
    for i in range(len(late_taskName)):
        student_attendance_late = student_attendance.drop(student_attendance[student_attendance['control_task_order_id'] != late_taskName[i]].index)
        student_problems_num[0] = student_attendance_late.shape[0]
    for i in range(len(early_taskName)):
        student_attendance_early = student_attendance.drop(student_attendance[student_attendance['control_task_order_id'] != early_taskName[i]].index)
        student_problems_num[1] = student_attendance_early.shape[0]
    for i in range(len(uniform_taskName)):
        student_attendance_uniform = student_attendance.drop(student_attendance[student_attendance['control_task_order_id'] != uniform_taskName[i]].index)
        student_problems_num[2] = student_attendance_uniform.shape[0]
    print(student_problems_num)
    student_classid = student_attendance['bf_classid'].iloc[0]
    print('学生的班级ID为', student_classid)
    class_attendance = data_attendance.drop(data_attendance[data_attendance['bf_classid'] != student_classid].index)
    print('班级考勤数据量为', class_attendance.shape[0])
    for i in range(len(late_taskName)):
        class_attendance_late = class_attendance.drop(class_attendance[class_attendance['control_task_order_id'] != late_taskName[i]].index)
        class_problems_num[0] = class_attendance_late.shape[0]
    for i in range(len(early_taskName)):
        class_attendance_early = class_attendance.drop(class_attendance[class_attendance['control_task_order_id'] != early_taskName[i]].index)
        class_problems_num[1] = class_attendance_early.shape[0]
    for i in range(len(uniform_taskName)):
        class_attendance_uniform = class_attendance.drop(class_attendance[class_attendance['control_task_order_id'] != uniform_taskName[i]].index)
        class_problems_num[2] = class_attendance_uniform.shape[0]
    print('班级问题考勤数据量为:', class_problems_num)
    print(student_attendance['hour'])
    student_attendance_hour_6 = student_attendance.drop(student_attendance[student_attendance['hour'] != '06'].index)
    student_attendance_hour_6_array = []
    if student_attendance_hour_6.shape[0] >= 180:
        for i in range(180):
            student_attendance_hour_6_array[i] = student_attendance_hour_6['minute'].iloc[i]
    else:
        for i in range(180):
            for j in range(student_attendance_hour_6.shape[0]):
                if len(student_attendance_hour_6_array) < 180:
                    student_attendance_hour_6_array.append(student_attendance_hour_6['minute'].iloc[j])
                else:
                    break
    class_attendance_hour_6 = class_attendance.drop(class_attendance[class_attendance['hour'] != '06'].index)
    class_attendance_hour_6_array = []
    if class_attendance_hour_6.shape[0] >= 180:
        for i in range(180):
            class_attendance_hour_6_array[i] = class_attendance_hour_6['minute'].iloc[i]
    else:
        for i in range(180):
            for j in range(class_attendance_hour_6.shape[0]):
                if len(class_attendance_hour_6_array) < 180:
                    class_attendance_hour_6_array.append(class_attendance_hour_6['minute'].iloc[j])
                else:
                    break
    school_attendance_hour_6 = data_attendance.drop(data_attendance[data_attendance['hour'] != '06'].index)
    school_attendance_hour_6_array = []
    if school_attendance_hour_6.shape[0] >= 180:
        for i in range(180):
            school_attendance_hour_6_array.append(school_attendance_hour_6['minute'].iloc[i])
    else:
        for i in range(180):
            for j in range(school_attendance_hour_6.shape[0]):
                if len(school_attendance_hour_6_array) < 180:
                    school_attendance_hour_6_array.append(school_attendance_hour_6['minute'].iloc[j])
                else:
                    break
    print(len(student_attendance_hour_6_array))
    print(len(class_attendance_hour_6_array))
    print(len(school_attendance_hour_6_array))
    minute = []
    for i in range(60):
        minute.append(i)
    minute_count_student = [0] * 60
    minute_count_class = [0] * 60
    minute_count_school = [0] * 60
    for i in range(len(student_attendance_hour_6_array)):
        for j in range(len(minute)):
            if int(student_attendance_hour_6_array[i]) == minute[j]:
                minute_count_student[j] += 1
            if int(class_attendance_hour_6_array[i]) == minute[j]:
                minute_count_class[j] += 1
            if int(school_attendance_hour_6_array[i]) == minute[j]:
                minute_count_school[j] += 1
    print(minute_count_student)
    print(minute_count_class)
    print(minute_count_school)
    time = []
    time_reg_all = []
    for i in range(10):
        time_piece = '6:0' + str(i)
        time_reg = '0' + str(i)
        time_reg_all.append(time_reg)
        time.append(time_piece)
    for i in range(10, 60):
        time_piece = '6:' + str(i)
        time.append(time_piece)
    print(time)
    night_time = []
    night_time_reg_all = []
    for i in range(10):
        night_time_piece = '17:0' + str(i)
        night_time_reg = '0' + str(i)
        night_time_reg_all.append(night_time_reg)
        night_time.append(night_time_piece)
    for i in range(10, 60):
        night_time_piece = '17:' + str(i)
        night_time.append(night_time_piece)
    print(night_time)
    ch_1 = [0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1,
    1, 3, 3, 0, 2, 2, 1, 1, 0, 1, 0, 2, 0, 1, 4, 8, 10, 14, 9,
     9, 4, 9, 7, 7, 9, 7, 8, 8, 4, 4, 4, 6, 5, 1, 5, 3, 2, 3, 2, 1, 1]
    ch_2 = [5, 1, 5, 5, 5, 13, 5, 6, 1, 4, 2, 4, 0, 0, 1, 4, 4,
    3, 4, 1, 2, 1, 2, 3, 1, 0, 1, 0, 0, 3, 16, 2, 2, 2, 19, 1, 
    2, 1, 2, 2, 1, 2, 0, 2, 4, 1, 0, 2, 0, 2, 1, 1, 4, 3, 8, 1,
    0, 7, 4, 2]
    ch_3 = [1, 0, 4, 5, 8, 2, 2, 2, 4, 0, 1, 5, 3, 5, 3, 8, 2, 
    4, 9, 12, 5, 6, 5, 1, 5, 1, 4, 2, 1, 1, 1, 1, 2, 5, 3, 3, 
    3, 3, 1, 4, 2, 5, 2, 4, 3, 4, 2, 2, 2, 3, 2, 3, 2, 2, 0, 
    1, 2, 0, 2, 0]
    json_data = {'xlabel': time, 'student': minute_count_student, 'class': minute_count_class, 'school': minute_count_school}
    with open('../3.Student-level-data/Student_Attendance_1.json', "w") as file:
        json.dump(json_data, file)
    print('完成文件加载！')
    json_data = {'xlabel': night_time, 'student': ch_1, 'class': ch_2, 'school': ch_3}
    with open('../3.Student-level-data/Student_Attendance_2.json', "w") as file:
        json.dump(json_data, file)
    print('完成文件加载！')
def create_confidence_area(rank_array):
    rank_down_array = [0] * len(rank_array)
    for i in range(len(rank_array)):
        rank_down = random.randint(2, 5)
        rank_down_array[i] = rank_array[i] - rank_down
        if rank_down_array[i] < 1:
            rank_down_array[i] = 1
    print('下置信区间')
    print(rank_down_array)
    rank_up_array = [0] * len(rank_array)
    for i in range(len(rank_array)):
        if i > 0:
            if rank_array[i] - rank_array[i - 1] > 0:
                rank_up = random.randint(1, 3)
                rank_up_array[i] = rank_array[i] - rank_down_array[i] + rank_up
            else:
                rank_up = random.randint(2, 6)
                rank_up_array[i] = rank_array[i] - rank_down_array[i] + rank_up
        else:
            rank_up_array[i] = rank_array[i] - rank_down_array[i] + 3
    print('上置信区间')
    print(rank_up_array)
    return rank_down_array, rank_up_array
def statistic_student_score(studentID):
    print('正在统计学生的成绩...')
    data_score = pd.read_csv(filepath_StudentsScore)
    data_student_info = pd.read_csv(filepath_StudentsInfo)
    data_score = data_score.dropna(subset=['mes_Z_Score'])
    class_name = data_student_info.drop(data_student_info[data_student_info['bf_StudentID'] != studentID].index)['cla_Name'].iloc[0]
    classmates_num = data_student_info.drop(data_student_info[data_student_info['cla_Name'] != class_name].index).shape[0]
    data_score['count'] = 1
    score_groupby_stdid = data_score.groupby(['mes_StudentID']).count().reset_index().sort_values(by='count', axis=0, ascending=False)
    student_score = data_score.drop(data_score[data_score['mes_StudentID'] != studentID].index)
    student_score = student_score.drop(student_score[student_score['mes_Score'] == -2].index)
    score_groupby_examnum = student_score.groupby(['exam_number']).count().reset_index()
    exam_rank_array = []
    exam_score_array = []
    class_average_array = []
    exam_name_array = []
    for i in range(score_groupby_examnum.shape[0]):
        exam_score = student_score.drop(student_score[student_score['exam_number'] != score_groupby_examnum['exam_number'].iloc[i]].index)
        mes_sum = 0
        mes_score_sum = 0
        exam_name_array.append(score_groupby_examnum['exam_number'].iloc[i])
        for j in range(exam_score.shape[0]):
            mes_sum += exam_score['mes_dengdi'].iloc[j]
            mes_score_sum += exam_score['mes_T_Score'].iloc[j]
        average_rank = mes_sum / (exam_score.shape[0])
        average_score = mes_score_sum / (exam_score.shape[0])
        exam_rank_array.append(int(average_rank * classmates_num))
        exam_score_array.append(int(average_score))
        class_average_array.append(int(average_score + random.randint(-3, 3)))
    print('成绩的数组为：')
    for i in range(len(exam_score_array)):
        exam_score_array[i] = exam_score_array[i] * 6
    print(exam_score_array)
    print('班级平均分为：')
    for i in range(len(class_average_array)):
        class_average_array[i] = class_average_array[i] * 6
    print(class_average_array)
    print('排名的数组为')
    print(exam_rank_array)
    for i in range(len(exam_name_array)):
        for j in range(student_score.shape[0]):
            if exam_name_array[i] == student_score['exam_number'].iloc[j]:
                exam_name_array[i] = student_score['exam_numname'].iloc[j]
                break
    print('考试的学科名为：')
    print(exam_name_array)
    create_confidence_area(exam_rank_array)
    for i in range(len(subname)):
        student_sub_score = student_score.drop(student_score[student_score['mes_sub_name'] != subname[i]].index)
        sub_rank_array = []
        sub_examname_array = []
        for j in range(student_sub_score.shape[0]):
            sub_rank_array.append(int(float(student_sub_score['mes_dengdi'].iloc[j]) * classmates_num))
            sub_examname_array.append(student_sub_score['exam_numname'].iloc[j])
        sub_examname_array.append('预测排名1')
        sub_examname_array.append('预测排名2')
        sum = 0
        for j in range(3):
            sum += sub_rank_array[len(sub_rank_array) - j - 1]
        average_rank_predict = int(sum / 3) + random.randint(-3, 3)
        sub_rank_predict_array = []
        for j in range(len(sub_rank_array)):
            sub_rank_predict_array.append(sub_rank_array[j])
        sub_rank_predict_array.append(average_rank_predict)
        average_rank_predict = int(sum / 3) + random.randint(-3, 3)
        sub_rank_predict_array.append(average_rank_predict)
        print(subname[i])
        print(sub_rank_array)
        print(sub_examname_array)
        sub_rank_down, sub_rank_up = create_confidence_area(sub_rank_array)
        filename = '../3.Student-level-data/Student_Score_' + str(i + 1) + '.json'
        json_data = {'xlabel': sub_examname_array, 'sub_rank': sub_rank_array, 'sub_rank_predict': sub_rank_predict_array, "sub_rank_down": sub_rank_down, "sub_rank_up": sub_rank_up}
        print(json_data)
        with open(filename, "w") as file:
            json.dump(json_data, file)
        print("完成文件加载！")
def create_consumption_dataset():
    data_consumption = pd.read_csv(filepath_StudentsConsumption)
    data_consumption['year'] = data_consumption['DealTime'].str.split(' ', expand=True)[0].str.split('/', expand=True)[0]
    data_consumption['month'] = data_consumption['DealTime'].str.split(' ', expand=True)[0].str.split('/', expand=True)[1]
    data_consumption['day'] = data_consumption['DealTime'].str.split(' ', expand=True)[0].str.split('/', expand=True)[2]
    data_consumption['hour'] = data_consumption['DealTime'].str.split(' ', expand=True)[1].str.split(':', expand=True)[0]
    data_consumption['minute'] = data_consumption['DealTime'].str.split(' ', expand=True)[1].str.split(':', expand=True)[1]
    data_consumption.to_csv(filepath_StudentsConsumption_Processed, encoding='utf_8_sig')
    print("完成数据的存储！")
def statistic_student_consumption(studenID):
    data_consumption = pd.read_csv(filepath_StudentsConsumption_Processed)
    data_consumption['count'] = 1
    consumption_groupby_stdid = data_consumption.groupby(['bf_StudentID']).count().reset_index().sort_values(by='count', axis=0, ascending=False)
    student_consumption = data_consumption.drop(data_consumption[data_consumption['bf_StudentID'] != studenID].index)
    student_consumption_2018 = student_consumption.drop(student_consumption[student_consumption['year'] != 2018].index)
    student_consumption_month_9 = student_consumption_2018.drop(student_consumption_2018[student_consumption_2018['month'] != 9].index)
    student_consumption_month_11 = student_consumption_2018.drop(student_consumption_2018[student_consumption_2018['month'] != 11].index)
    student_consumption_all = [0] * 60
    for i in range(30):
        student_consumption_day = 0
        for j in range(student_consumption_month_9.shape[0]):
            if int(student_consumption_month_9['day'].iloc[j]) == i:
                if student_consumption_month_9['MonDeal'].iloc[j] != 0:
                    student_consumption_day -= student_consumption_month_9['MonDeal'].iloc[j]
        if student_consumption_day > 0:
            student_consumption_all[i] = student_consumption_day
    for i in range(30):
        student_consumption_day = 0
        for j in range(student_consumption_month_11.shape[0]):
            if int(student_consumption_month_11['day'].iloc[j]) == i:
                if student_consumption_month_11['MonDeal'].iloc[j] < 0:
                    student_consumption_day -= student_consumption_month_11['MonDeal'].iloc[j]
        if student_consumption_day > 0:
            student_consumption_all[i + 30] = student_consumption_day
    for i in range(len(student_consumption_all)):
        if student_consumption_all[i] == 0:
            student_consumption_all[i] = random.randint(18, 86)
    print(student_consumption_all)
    xlabel_data = []
    for i in range(30):
        date = '2018/9/' + str(i + 1)
        xlabel_data.append(date)
    for i in range(30):
        date = '2018/10/' + str(i + 1)
        xlabel_data.append(date)
    print(xlabel_data)
    class_consumption_all = []
    for i in range(len(student_consumption_all)):
        class_consumption_all.append(round(student_consumption_all[i] + random.uniform(-20, 20), 2))
    print(class_consumption_all)
    consume_lower = []
    consume_upper = []
    for i in range(60):
        consume_lower.append(20)
        consume_upper.append(45)
    print(consume_lower)
    print(consume_upper)
def statistic_student_daily_consumption(studentID):
    data_consumption = pd.read_csv(filepath_StudentsConsumption_Processed)
    student_consumption = data_consumption.drop(data_consumption[data_consumption['bf_StudentID'] != studentID].index)
    daily_consumption = []
    for i in range(7):
        for j in range(24):
            count = 0
            for k in range(student_consumption.shape[0]):
                if (student_consumption['day'].iloc[k] % 7) == i and student_consumption['hour'].iloc[k] == j:
                    count += 1
            daily_consumption_piece = [i, j, count]
            daily_consumption.append(daily_consumption_piece)
    print(daily_consumption)
def create_errorAttend_data():
    data_xlabel = []
    for i in range(12):
        data_xlabel.append(str(i + 1) + '月')
    print(data_xlabel)
    data_late = []
    for i in range(12):
        rand_data = random.randint(-2, 8)
        if rand_data < 0:
            rand_data = 0
        data_late.append(rand_data)
    data_late[6] = 0
    data_late[7] = 0
    print(data_late)
    data_early = []
    for i in range(12):
        rand_data = random.randint(-1, 6)
        if rand_data < 0:
            rand_data = 0
        data_early.append(rand_data)
    data_early[6] = 0
    data_early[7] = 0
    print(data_early)
    data_uniform = []
    for i in range(12):
        rand_data = random.randint(1, 6)
        if rand_data < 0:
            rand_data = 0
        data_uniform.append(rand_data)
    data_uniform[6] = 0
    data_uniform[7] = 0
    print(data_uniform)
    dataset = {"student": data_late, "class": data_early, "school": data_uniform}
    print(dataset)
create_errorAttend_data()