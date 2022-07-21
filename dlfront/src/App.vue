<template>
  <el-container class="layout-container-demo" style="height: 500px">
    <el-aside width="220px">
      <el-scrollbar>
        <!-- 选择数据集 -->
        <el-button type="primary" plain style="margin: 20px" @click="drawer1 = true" size="large">

          <el-icon style="width: 1em; height: 1em; margin: 20px">
            <message />
          </el-icon>选择数据集
        </el-button>

        <el-drawer v-model="drawer1" :direction="direction">
          <template #default>
            <el-radio-group v-model="datasetName">
              <el-radio v-model="radio1" v-for="(name, idx) in datasetList" :key="name" :label="name">{{ name }}
              </el-radio>
             
            </el-radio-group>
          </template>
          <template #footer>
            <div style="flex: auto">
              <el-button @click="cancelClick1">cancel</el-button>
              <el-button type="primary" @click="confirmClick1">confirm</el-button>
            </div>
          </template>
        </el-drawer>


        <!-- 数据切分器 -->

        <el-button type="primary" plain style="margin: 20px" @click="drawer2 = true" size="large">

          <el-icon style="width: 1em; height: 1em; margin: 20px">
            <setting />
          </el-icon>选择数据切分器
        </el-button>

        <el-drawer v-model="drawer2" :direction="direction">
          <template #default>
            <el-radio-group v-model="splitterName">
              <el-radio v-model="radio2" v-for="(name, idx) in splitterList" :key="name" :label="name">{{ name }}
              </el-radio>
            </el-radio-group>
          </template>
          <template #footer>
            <div style="flex: auto">
              <el-button @click="cancelClick2">cancel</el-button>
              <el-button type="primary" @click="confirmClick2">confirm</el-button>
            </div>
          </template>
        </el-drawer>
        <!-- 选择模型 -->

        <el-button type="primary" plain style="margin: 20px" @click="drawer3 = true" size="large">

          <el-icon style="width: 1em; height: 1em; margin: 20px">
            <icon-menu />
          </el-icon>选择模型
        </el-button>

        <el-drawer v-model="drawer3" :direction="direction">
          <template #default>
            <el-radio-group v-model="modelName">
              <el-radio v-model="radio3" v-for="(name, idx) in modelList" :key="name" :label="name">{{ name }}
              </el-radio>
            </el-radio-group>
          </template>
          <template #footer>
            <div style="flex: auto">
              <el-button @click="cancelClick3">cancel</el-button>
              <el-button type="primary" @click="confirmClick3">confirm</el-button>
            </div>
          </template>
        </el-drawer>

        <!-- 选择判别器 -->

        <el-button type="primary" plain style="margin: 20px" @click="drawer4 = true" size="large">

          <el-icon style="width: 1em; height: 1em; margin: 20px">
            <setting />
          </el-icon>选择判别器
        </el-button>

        <el-drawer v-model="drawer4" :direction="direction">
          <template #default>
            <div>
              <el-radio-group v-model="judgerName">
                <el-radio v-model="radio4" v-for="(name, idx) in judgerList" :key="name" :label="name">{{ name }}
                </el-radio>
              </el-radio-group>
            </div>
          </template>
          <template #footer>
            <div style="flex: auto">
              <el-button @click="cancelClick4">cancel</el-button>
              <el-button type="primary" @click="confirmClick4">confirm</el-button>
            </div>
          </template>
        </el-drawer>


        <!-- 可以采用树形选择器来表示一个数据集在不同算法下的结果 -->


      </el-scrollbar>
    </el-aside>

    <el-container>
      <el-header style="text-align: right; font-size: 12px">

        <div class="header">
          <!-- <el-button type="primary" disabled plain style="position:absolute ; left:60% ;top:22%  ">结果显示</el-button> -->
          <el-icon style="size: 10px; position:absolute ; left:48% ;top:40%" color=#87CEEB>
            <icon-menu />
          </el-icon>
          <el-select style="position:absolute ; left:50% ;top:22%" clearable placeholder="结果展示">
            <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value" text
              @click="table = true" />
          </el-select>
          <el-drawer v-model="table" title="I have a nested table inside!" direction="rtl" size="50%">
            <el-table :data="gridData">
              <!-- <el-table-column property="date" label="Date" width="150" />
              <el-table-column property="name" label="Name" width="200" />
              <el-table-column property="address" label="Address" /> -->
            
            </el-table>
          </el-drawer>


        </div>
        <div class="header">
          <el-button type="primary" plain style="position:absolute ; right:75% ;top:22%  " @click="clickButton">执行算法
          </el-button>
        </div>
         
        <div class="toolbar">
          <el-dropdown>
            <el-icon style="margin-right: 8px; margin-top: 1px">
              <setting />
            </el-icon>
            <template #dropdown>
              <el-dropdown-menu>

                <el-dropdown-item text @click="add">Add</el-dropdown-item>
                <el-dropdown-item>Delete</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>

        </div>


      </el-header>
      <el-row :gutter="20">
        <el-col :span="12">
          <div class="grid-content ep-bg-purple" />
          <p></p>
          <p></p>
          <div class="running-res" >
            <div v-for="(content, idx) in runningOutput" :key="idx" border:2px solid black>{{ content
            }}
            </div>


          </div>
          <!--  <headers title="新建记录"></headers> -->
          <!-- <el-input v-model="textarea" :rows="10" type="textarea" placeholder="Please input" /> -->

        </el-col>
        <el-col :span="12">
          <div class="grid-content ep-bg-purple-light" >
          <p></p>
          <p></p>
          
          
           <img v-if="flag2" alt="鸢尾花-KNN" src="./assets/test_iris_KNN9.png" />
           <img v-if="flag3" alt="波士顿XGB" src="./assets/波士顿房价-测试集XGB10.png" />
           <img v-if="flag4" alt="波士顿SVR" src="./assets/波士顿房价-测试集SVR10.png" />
           <img v-if="flag5" alt="波士顿XGB特征" src="./assets/波士顿房价XGB重要特征10.png" />
           <img v-if="flag6" alt="鸢尾花GBM" src="./assets/鸢尾花-测试集GBM10.png" />
        
          </div>


          <!-- <div class="demo-image__preview">
            <el-image  class="image" style="width: 100px; height: 100px" :src="url" :preview-src-list="srcList"
              :initial-index="4" fit="cover" />
          </div> -->

        </el-col>
      </el-row>

    </el-container>
  </el-container>
  <el-dialog v-model="showConnectInfoWindow" title="连接地址" :show-close="false">
    <el-form label-width="40px">
      <el-form-item label="Url">
        <el-input v-model="connectUrl" />
      </el-form-item>
      <el-form-item label="Port">
        <el-input v-model="connectPort" />
      </el-form-item>
    </el-form>
    <el-button type="primary" @click="onClickConnect">连接</el-button>
  </el-dialog>

</template>

<script lang="ts" setup>


import { ref } from 'vue'
import { Menu as IconMenu, Message, Setting } from '@element-plus/icons-vue'
import { h } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { requiredNumber } from 'element-plus/es/components/table-v2/src/common';
const table = ref(false)
const drawer2 = ref(false)
const drawer3 = ref(false)
const drawer4 = ref(false)
const drawer1 = ref(false)
const isshow=false
const direction = ref('ttb')
const radio1 = ref()
const radio2 = ref()
const radio3 = ref()
const radio4 = ref()
const flag2=ref(false)
const flag3=ref(false)
const flag4=ref(false)
const flag5=ref(false)
const flag6=ref(false)
const gridData = [
  {}
]
const options = [
  {
    value: 'Option1',
    label: 'Option1',
  },
  {
    value: 'Option2',
    label: 'Option2',
  },
  {
    value: 'Option3',
    label: 'Option3',
  },
]

/* const jpg= require (".../public/训练集.png") */

/* function getData() {
  //  更新数据market_id.txt文件接口
  let xhr = new XMLHttpRequest(),
    okStatus = document.location.protocol === "file:" ? 0 : 200;
  xhr.open("GET", "../../KNN.txt", false); // 文件路径
  xhr.overrideMimeType("text/html;charset=utf-8"); //默认为utf-8
  xhr.send(null);
  console.log(xhr.responseText) //文本的内容
}
function mounted() {
  getData();
} */



const handleClose = (done: () => void) => {
  ElMessageBox.confirm('Are you sure you want to close this?')
    .then(() => {
      done()
    })
    .catch(() => {
      // catch error
    })
}
function cancelClick1() {

  drawer1.value = false
}
function confirmClick1() {
  ElMessageBox.confirm(`Are you confirm to chose ${radio1.value} ?`)
    .then(() => {
      drawer1.value = false
    })
    .catch(() => {
      // catch error
    })
}
function cancelClick2() {

  drawer2.value = false
}
function confirmClick2() {
  ElMessageBox.confirm(`Are you confirm to chose ${radio2.value} ?`)
    .then(() => {
      drawer2.value = false
    })
    .catch(() => {
      // catch error
    })
}
function cancelClick3() {

  drawer3.value = false
}
function confirmClick3() {
  ElMessageBox.confirm(`Are you confirm to chose ${radio3.value} ?`)
    .then(() => {
      drawer3.value = false
    })
    .catch(() => {
      // catch error
    })
}
function cancelClick4() {

  drawer4.value = false
}
function confirmClick4() {
  ElMessageBox.confirm(`Are you confirm to chose ${radio4.value} ?`)
    .then(() => {
      drawer4.value = false
    })
    .catch(() => {
      // catch error
    })
}

/* function page(){

}
 */


const connectUrl = ref('localhost')
const connectPort = ref('8765')
const showConnectInfoWindow = ref(false)

const consequnce = () => {

}

let ws: WebSocket | null = null
const connect = () => {
  ws = new WebSocket('ws://' + connectUrl.value + ':' + connectPort.value)
  ws.onopen = () => {
    isConnectedToServer.value = true
    showConnectInfoWindow.value = false
    ElMessage({
      showClose: true,
      message: '连接成功',
      type: 'success',
    })
    ws?.send(JSON.stringify({
      'type': 'overview',
      'params': {}
    }))
  }
  ws.onmessage = (evt) => {
    var received_msg = JSON.parse(evt.data);
    // console.log(received_msg);

    if (received_msg.status === 200) {
      const received_msg_data = received_msg.data
      if (received_msg.type === 'overview') {
        datasetList.value = received_msg_data.datasets
        splitterList.value = received_msg_data.splitters
        modelList.value = received_msg_data.models
        judgerList.value = received_msg_data.judgers
      }

      else if (received_msg.type === 'print') {
        runningOutput.value.push(received_msg_data.content as never)
      }
    } else {
      console.error(received_msg.data);
    }
  }
  ws.onclose = () => {
    isConnectedToServer.value = false
    showConnectInfoWindow.value = true
    ElMessage({
      showClose: true,
      message: '连接已断开',
      type: 'error'
    })
  }
  ws.onerror = () => {
    ElMessage({
      showClose: true,
      message: '连接失败(地址错误 / 协议错误 / 服务器错误)',
      type: 'error'
    })
    showConnectInfoWindow.value = true
  }
}

connect()

const datasetName = ref('')
const datasetList = ref([])

const splitterName = ref(null)
const splitterList = ref([])

const modelName = ref('')
const modelList = ref([])

const judgerName = ref(null)
const judgerList = ref([])

const txt = ref(null)

const isConnectedToServer = ref(false)
const onClickConnect = () => {
  connect()
}

const clickButton = () => {
if(datasetName.value=='鸢尾花' && modelName.value=='KNN')
{
  flag2.value=true
  flag3.value=false
  flag4.value=false
  flag5.value=false
  flag6.value=false
}
if(datasetName.value=='鸢尾花' && modelName.value=='GBM')
{
  flag2.value=false
  flag3.value=false
  flag4.value=false
  flag5.value=false
  flag6.value=true
}
else if(datasetName.value=='波士顿房价' && modelName.value=='XGB')
{
  flag2.value=false
  flag3.value=true
  flag4.value=false
  flag5.value=true
  flag6.value=false
}
else if(datasetName.value=='波士顿房价' && modelName.value=='SVR')
{
  flag2.value=false
  flag3.value=false
  flag4.value=true
  flag5.value=false
  flag6.value=false
}


  if (!isConnectedToServer.value) {
    ElMessage({
      showClose: true,
      message: '与 server 的连接已断开。请重启 python 服务后刷新页面',
      type: 'error'
    })
    return
  }
  if (!datasetName.value || !splitterName.value || !modelName.value || !judgerName.value) {

    ElMessage(
      {
        showClose: true,
        message: '您的选项不完整',
        type: 'error'
      })

  }
  
  runningOutput.value = []
/*  this.isshow=!this.isshow */
  ws?.send(JSON.stringify({
    'type': 'run',
    'params': {
      'datasetName': datasetName.value,
      'splitterName': splitterName.value,
      'modelName': modelName.value,
      'judgerName': judgerName.value
    }
  }))
  
}

const add = () => {
  ElMessageBox({
    title: 'add',
    message: h('p', null, [
      h('span', null, 'Message can be '),
      h('i', { style: 'color: teal' }, 'VNode'),
    ]),
    showCancelButton: true,
    confirmButtonText: 'OK',
    cancelButtonText: 'Cancel',
    beforeClose: (action, instance, done) => {
      if (action === 'confirm') {
        instance.confirmButtonLoading = true
        instance.confirmButtonText = 'Loading...'
        setTimeout(() => {
          done()
          setTimeout(() => {
            instance.confirmButtonLoading = false
          }, 300)
        }, 3000)
      } else {
        done()
      }
    },
  }).then((action) => {
    ElMessage({
      type: 'info',
      message: `action: ${action}`,
    })
  })
}
const runningOutput = ref([])
const textarea = ref('')
</script>


<style scoped>
.layout-container-demo .el-header {
  position: relative;
  background-color: var(--el-color-primary-light-7);
  color: var(--el-text-color-primary);
}

.layout-container-demo .el-aside {
  color: var(--el-text-color-primary);
  background: var(--el-color-primary-light-8);
}

.layout-container-demo .el-menu {
  border-right: none;
}

.layout-container-demo .el-main {
  padding: 0;
}

.layout-container-demo .toolbar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  right: 20px;
}

.running-res {
  text-align: left;
  width: 100%;
  font-size: large;

}
</style>
