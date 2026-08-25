# 数据目录契约

原始数据不进入Git。服务器默认从 `/data/aiot/raw` 读取以下文件：

```text
README.docx
03prru_cw_position.csv
01参考标签信息/df_train_lib_LOS.csv
01参考标签信息/df_train_lib_NLOS.csv
04场景点云/成都餐厅点云/8_21_2026/AIOT_scene.ply
```

程序读取时固定执行 `rsrp_6/64` 和 `(x,y,z)_map=(x,-y,z)_file`，不要修改原始CSV。

