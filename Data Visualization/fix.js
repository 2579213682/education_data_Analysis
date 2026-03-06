const fs = require('fs');
const path = require('path');

const files = [
    'src/components/ApplicationDetailCj.js',
    'src/components/ApplicationDetailCsp.js',
    'src/components/ApplicationDetailKq.js',
    'src/components/ApplicationDetailStu.js',
    'src/components/ApplicationDetailTeacher.js'
];

files.forEach(file => {
    const filePath = path.join(__dirname, file);
    if (fs.existsSync(filePath)) {
        let content = fs.readFileSync(filePath, 'utf8');

        // 检查是否有 filterIcon 使用了 Icon
        if (content.includes('filterIcon') && content.includes('<Icon')) {

            // 添加 SearchOutlined 导入（如果还没有）
            if (!content.includes('SearchOutlined')) {
                // 找到 @ant-design/icons 导入行
                if (content.includes("@ant-design/icons")) {
                    // 在现有导入中添加 SearchOutlined
                    content = content.replace(
                        /from\s+['"]@ant-design\/icons['"];/,
                        `from '@ant-design/icons';`
                    );
                    // 在文件顶部添加新的导入行
                    content = `import { SearchOutlined } from '@ant-design/icons';\n` + content;
                } else {
                    // 添加新的导入
                    content = `import { SearchOutlined } from '@ant-design/icons';\n` + content;
                }
            }

            // 替换 filterIcon: <Icon type="search" /> 或类似用法
            content = content.replace(
                /filterIcon:\s*<Icon[^>]*type=["']search["'][^>]*\/>/g,
                'filterIcon: <SearchOutlined />'
            );

            // 替换 filterIcon: filtered => <Icon ... />
            content = content.replace(
                /filterIcon:\s*filtered\s*=>\s*<Icon[^>]*\/>/g,
                'filterIcon: filtered => <SearchOutlined style={{ color: filtered ? "#1890ff" : undefined }} />'
            );

            // 替换其他 Icon 用法
            content = content.replace(
                /<Icon([^>]*)\/>/g,
                '<SearchOutlined$1 />'
            );

            fs.writeFileSync(filePath, content);
            console.log(`✅ 已修复: ${file}`);
        } else {
            console.log(`⏭️  跳过: ${file} (无需修复)`);
        }
    } else {
        console.log(`❌ 未找到: ${file}`);
    }
});

console.log('\n修复完成！请运行 npm start 查看效果');