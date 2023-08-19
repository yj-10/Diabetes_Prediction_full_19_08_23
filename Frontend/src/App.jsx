import { Form, Slider } from "antd";
import TableChart from "./Component/TableChart";
import "./App.css";
import React, { useEffect, useState } from "react";
import { Col, Container, Row } from "react-bootstrap";
import infosense from "../src/assest/Infosense.png";
import LineChart from "./Component/LineChart";
import Footer from "./Component/Footer";
import * as mark from "./Component/marksinfo";
import { BsTelephone } from "react-icons/bs";
import axios from "axios";
import Table from "react-bootstrap/Table";
// import Aform from "../src/Component/Aform";
function App() {
  const [showdata, setdata] = useState("");
  const [ApiData, setApiData] = useState({});
  console.log(ApiData.result);
  console.log("452", showdata);
  const onFinish = async (values) => {
    let formData = new FormData();
    formData.append("pregnancies", values.pregnancies);
    formData.append("glucose", values.glucose);
    formData.append("bloodPressure", values.bloodPressure);
    formData.append("skinThickness", values.skinThickness);
    formData.append("insulin", values.insulin);
    formData.append("BMI", values.BMI);
    formData.append(
      "diabetesPedigreeFunction",
      values.diabetesPedigreeFunction
    );
    formData.append("Age", values.Age);

    // await axios
    //   .post("http://127.0.0.1:8000/core/get_diabetes_data", { values })
    //   .then(function (response) {
    //     console.log(response);
    //   });

    // setdata(values);
    setdata(values);
    const response = await axios({
      method: "post",
      url: "http://127.0.0.1:8000/core/get_diabetes_data",
      data: formData,
    })
      .then((resp) => {
        if (resp.data.success === true) {
          setApiData(resp.data.response);
        } else if (resp.data.status == 400) {
          console.log("response data status 4000");
        }
      })
      .catch((error) => {
        console.log("error", error.message);
      });

    console.log("Received values of form: ", values);
  };

  return (
    <>
      <div className="wrapper">

      {/* <Aform/> */}
        <div className="headers">
          {" "}
          <Container>
            <Row
              className="d-flex justify-content-between align-items-center"
              style={{ padding: "10px 0px" }}
            >
              <Col lg={8} md={6} sm={12}>
                <img src={infosense} className="img-fluid w-25" alt="no-img" />
              </Col>
              <Col lg={4} md={6} sm={12}>
                <div style={{ marginTop: "12px", textAlign: "end" }}>
                  <p>
                    <BsTelephone /> Contact Us :- +91 - 9343017165
                  </p>
                </div>
              </Col>
            </Row>
          </Container>
        </div>

        <div className="Marquee_container">
          <marquee direction="right" className="h6 text-uppercase">
            diabetes data analytics By Infosense
          </marquee>
        </div>

        <Container>
          <div className="Wrapper_title">
            <h1 className="text-center">Diabetes Prediction</h1>
          </div>
          <Form onFinish={onFinish}>
            <Row className="gx-5 ">
              <Col className="Row_col ">
                <div className="Wrapper_Content">
                  <Form.Item name="pregnancies" label="Pregnancies">
                    <Slider
                      marks={{
                        0: "0",
                        2: "2",
                        4: "4",
                        6: "6",
                        8: "8",
                        10: "10",
                        12: "12",
                        14: "14",
                        17: "17",
                      }}
                      min={0}
                      max={17}
                    />
                  </Form.Item>
                </div>
                <div className="Wrapper_Content  ">
                  <Form.Item name="glucose" label="Glucose">
                    <Slider
                      className="w-100"
                      marks={{
                        0: "0",
                        20: "20",
                        40: "40",
                        60: "60",
                        80: "80",
                        100: "100",
                        120: "120",
                        140: "140",
                        160: "160",
                        180: "180",
                        200: "200",
                      }}
                      min={0}
                      max={200}
                    />
                  </Form.Item>
                </div>
                <div className="Wrapper_Content">
                  <Form.Item name="bloodPressure" label="Blood Pressure">
                    <Slider
                      marks={{
                        0: "0",
                        20: "20",
                        40: "40",
                        60: "60",
                        80: "80",
                        100: "100",
                        122: "122",
                      }}
                      min={0}
                      max={122}
                    />
                  </Form.Item>
                </div>
                <div className="Wrapper_Content">
                  <Form.Item name="skinThickness" label="Skin Thikness">
                    <Slider
                      marks={{
                        0: "0",
                        20: "20",
                        40: "40",
                        60: "60",
                        80: "80",
                        100: "100",
                      }}
                      min={0}
                      max={100}
                    />
                  </Form.Item>
                </div>
              </Col>
              <Col lg={6} sm={12} md={6} className="Row_col ">
                <div className="Wrapper_Content">
                  <Form.Item name="insulin" label="Insulin">
                    <Slider
                      marks={{
                        0: "0",
                        200: "200",
                        400: "400",
                        600: "600",
                        700: "700",
                        846: "846",
                      }}
                      min={0}
                      max={846}
                    />
                  </Form.Item>
                </div>
                <div className="Wrapper_Content">
                  <Form.Item name="BMI" label="BMI">
                    <Slider
                      marks={{
                        0: "0",
                        20: "20",
                        40: "40",
                        67: "67",
                      }}
                      min={0}
                      max={67}
                    />
                  </Form.Item>
                </div>
                <div className="Wrapper_Content">
                  <Form.Item name="Age" label="Age">
                    <Slider
                      marks={{
                        0: "0",
                        20: "20",
                        40: "40",
                        60: "60",
                        88: "88",
                      }}
                      min={0}
                      max={88}
                    />
                  </Form.Item>
                </div>
                <div className="Wrapper_Content">
                  <Form.Item
                    name="diabetesPedigreeFunction"
                    label="Diabetes Pedigree"
                  >
                    <Slider
                      marks={{
                        0: "0",
                        1: "1",
                        2: "2",
                        3: "3",
                      }}
                      min={0}
                      max={3}
                      // step={0.01}
                    />
                  </Form.Item>
                </div>
              </Col>
            </Row>

            <div className="div_button my-4">
              <button>submit</button>
            </div>
          </Form>
        </Container>
        <div className="Wrapper_sec_Container">
          <Container>
            <Row className="gx-5 mb-5 ">
              <Col md={12} sm={12} lg={6}>
                <Table striped responsive bordered>
                  <thead>
                    <tr>
                      <th>Pregnancies</th>
                      <th>Glucose</th>
                      <th>Blood Pressure</th>
                      <th>Skin Thikness</th>
                      <th>Insulin</th>
                      <th>BMI</th>
                      <th>Age</th>
                      <th>Diabetes Pedigree </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{showdata == "" ? "0" : showdata.pregnancies}</td>
                      <td>{showdata == "" ? "0" : showdata.glucose}</td>
                      <td>{showdata == "" ? "0" : showdata.bloodPressure}</td>
                      <td>{showdata == "" ? "0" : showdata.skinThickness}</td>
                      <td>{showdata == "" ? "0" : showdata.insulin}</td>
                      <td>{showdata == "" ? "0" : showdata.BMI}</td>
                      <td>{showdata == "" ? "0" : showdata.Age}</td>
                      <td>
                        {showdata == ""
                          ? "0"
                          : showdata.diabetesPedigreeFunction}
                      </td>
                    </tr>
                  </tbody>
                </Table>
              </Col>
              <Col md={12} sm={12} lg={6}>
                <div className="Result_div border">
                  <div className="Result_Row d-flex justify-content-start align-items-center">
                    <article>
                      <h2 className="py-4"> Result : {ApiData.result}</h2>
                      {/* <h2> Model Accuracy : {ApiData.accuracy}</h2> */}
                    </article>
                  </div>
                </div>
              </Col>
            </Row>
          </Container>
        </div>
      </div>
      {/* Footer */}
      <Footer />
    </>
  );
}

export default App;

// import React from 'react'
// import Aform from "../src/Component/Aform";

// function App() {
//   return (
//     <>


// <Aform/>


//     </>
//   )
// }

// export default App
